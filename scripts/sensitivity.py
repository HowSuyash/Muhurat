"""Does the arm ranking survive our own assumptions being wrong?

THE OBJECTION THIS ANSWERS
--------------------------
"Your six world constants are invented, so your numbers are invented."

That is fair. `p_in_retry = 0.90` is an informed estimate, not a measurement.
But the submission does not claim those numbers are right — it claims the
*ranking* of arms is a real finding. This script tests exactly that claim by
perturbing every constant and checking whether the ordering survives.

WHAT IS PERTURBED
-----------------
Five of the six constants act only at execution time and can be varied without
touching the corpus:

    p_in_retry, p_in_rail, p_in_contact, p_out, attrition_lambda

The sixth, `attrition_horizon_days`, is baked into the hidden windows at corpus
generation, so varying it requires regenerating the corpus AND the ground truth.
That is done too, in its own pass, because a horizon-only sweep is the one a
sceptic would ask for first: it directly controls how much waiting is punished.

WHAT COUNTS AS SURVIVING
------------------------
Two claims are checked on every scenario:

  1. `llm_recommended` > `rules_recommended`     -- the headline finding
  2. `max_wait_probe` is not first               -- the degeneracy guarantee

A scenario that breaks either is reported, not hidden. Absolute rupee figures
move a lot under +/-40%; if the ordering moves too, the finding is fragile and
the README should say so.

Usage:
    python -m scripts.sensitivity              # +/-40% on each constant, one at a time
    python -m scripts.sensitivity --joint 12   # also 12 random joint perturbations
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

from app.corpus.generator import (
    DEFAULT_CORPUS_DIR,
    corpus_path_for,
    generate_corpus,
    read_corpus,
    write_corpus,
)
from app.corpus.truth import (
    WorldConstants,
    ground_truth_path_for,
    load_world,
    write_ground_truth,
)
from app.diagnosis.classifier import RulesClassifier
from app.diagnosis.llm_classifier import LLMClassifier
from app.execution.window import RecoveryWindowExecutor
from app.orchestrator import run_arm
from app.policy.backoff_skip import BackoffSkipPolicy
from app.policy.max_wait import MaxWaitPolicy
from app.policy.naive import NaiveRetryPolicy
from app.policy.payday_inference import PaydayInferencePolicy
from app.policy.rules_recommended import RulesRecommendedPolicy

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "data" / "runs" / "_sensitivity"

ARMS = {
    "naive_retry_3x": (NaiveRetryPolicy, "rules"),
    "backoff_skip": (BackoffSkipPolicy, "rules"),
    "rules_recommended": (RulesRecommendedPolicy, "rules"),
    "llm_recommended": (RulesRecommendedPolicy, "llm"),
    "payday_inference": (PaydayInferencePolicy, "rules"),
    "max_wait_probe": (MaxWaitPolicy, "rules"),
}

#: Constants that act at execution time only.
EXEC_CONSTANTS = ("p_in_retry", "p_in_rail", "p_in_contact", "p_out", "attrition_lambda")


def score_all(payments, truth_path, world, seed: int) -> dict[str, int]:
    """Recovered paise per arm under one world. No audit DB, no disk churn."""
    executor = RecoveryWindowExecutor(truth_path, world=world)
    rules = RulesClassifier()
    llm = LLMClassifier(rules=rules, allow_api=False)
    out: dict[str, int] = {}
    for name, (policy_cls, kind) in ARMS.items():
        cap = 1 if name == "max_wait_probe" else 3
        _, records = run_arm(
            payments,
            classifier=llm if kind == "llm" else rules,
            policy=policy_cls(max_attempts=cap),
            executor=executor,
            seed=seed,
            corpus_path=truth_path,
            arm=name,
            run_id=f"sens-{name}",
            runs_dir=SCRATCH,
            enable_sqlite=False,
        )
        out[name] = sum(r.amount_recovered_paise for r in records)
    return out


def verdict(scores: dict[str, int]) -> tuple[bool, bool, str]:
    ranked = sorted(scores, key=lambda k: -scores[k])
    llm_wins = scores["llm_recommended"] > scores["rules_recommended"]
    probe_loses = ranked[0] != "max_wait_probe"
    return llm_wins, probe_loses, ranked[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Sensitivity sweep over the world constants.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--delta", type=float, default=0.40, help="fractional perturbation (default 0.40)")
    ap.add_argument("--joint", type=int, default=0, help="also run N random joint perturbations")
    ap.add_argument("--json", type=Path, default=None, help="also write the scenarios as JSON")
    args = ap.parse_args()

    SCRATCH.mkdir(parents=True, exist_ok=True)
    corpus_path = corpus_path_for(args.seed, args.n)
    truth_path = ground_truth_path_for(args.seed, args.n, DEFAULT_CORPUS_DIR)
    payments = read_corpus(corpus_path)
    base_world = load_world()

    def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, v))

    rows: list[tuple[str, dict[str, int]]] = []

    base = score_all(payments, truth_path, base_world, args.seed)
    rows.append(("baseline", base))

    # --- one constant at a time -------------------------------------------
    for const in EXEC_CONSTANTS:
        for direction in (-1, 1):
            factor = 1.0 + direction * args.delta
            current = getattr(base_world.constants, const)
            hi = 10.0 if const == "attrition_lambda" else 1.0
            perturbed = replace(base_world.constants, **{const: clamp(current * factor, 0.0, hi)})
            world = replace(base_world, constants=perturbed)
            label = f"{const} {'+' if direction > 0 else '-'}{int(args.delta * 100)}%"
            rows.append((label, score_all(payments, truth_path, world, args.seed)))

    # --- the horizon, which needs the corpus rebuilt ----------------------
    for direction in (-1, 1):
        factor = 1.0 + direction * args.delta
        days = base_world.constants.attrition_horizon_days * factor
        world = replace(base_world, constants=replace(base_world.constants, attrition_horizon_days=days))
        pays, truths = generate_corpus(n=args.n, seed=args.seed, world=world)
        cp = SCRATCH / "corpus.jsonl"
        tp = SCRATCH / "truth.jsonl"
        write_corpus(pays, cp)
        write_ground_truth(truths, tp)
        label = f"attrition_horizon_days {'+' if direction > 0 else '-'}{int(args.delta * 100)}%"
        rows.append((label, score_all(pays, tp, world, args.seed)))

    # --- joint random perturbations ---------------------------------------
    if args.joint:
        rnd = random.Random(args.seed)
        for i in range(args.joint):
            kw = {}
            for const in EXEC_CONSTANTS:
                f = 1.0 + rnd.uniform(-args.delta, args.delta)
                hi = 10.0 if const == "attrition_lambda" else 1.0
                kw[const] = clamp(getattr(base_world.constants, const) * f, 0.0, hi)
            world = replace(base_world, constants=replace(base_world.constants, **kw))
            rows.append((f"joint #{i + 1}", score_all(payments, truth_path, world, args.seed)))

    # --- report ------------------------------------------------------------
    print()
    print("SENSITIVITY SWEEP  --  does the ranking survive our assumptions being wrong?")
    print("=" * 104)
    hdr = f"  {'scenario':<34} {'llm':>11} {'rules':>11} {'delta':>10} {'llm>rules':>10} {'winner':>18}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    llm_wins_n = probe_loses_n = 0
    for label, scores in rows:
        llm_wins, probe_loses, winner = verdict(scores)
        llm_wins_n += llm_wins
        probe_loses_n += probe_loses
        delta = (scores["llm_recommended"] - scores["rules_recommended"]) / 100
        print(
            f"  {label:<34} {scores['llm_recommended'] / 100:>11,.0f}"
            f" {scores['rules_recommended'] / 100:>11,.0f} {delta:>10,.0f}"
            f" {'yes' if llm_wins else 'NO':>10} {winner:>18}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "delta": args.delta,
            "scenarios": [
                {
                    "label": label,
                    "scores": {k: v for k, v in sc.items()},
                    "llm_wins": verdict(sc)[0],
                    "probe_loses": verdict(sc)[1],
                    "winner": verdict(sc)[2],
                }
                for label, sc in rows
            ],
        }, separators=(",", ":")), encoding="utf-8")
        print()
        print(f"  wrote {args.json}")

    total = len(rows)
    print()
    print(f"  llm_recommended beats rules_recommended in {llm_wins_n}/{total} scenarios")
    print(f"  max_wait_probe never wins in                {probe_loses_n}/{total} scenarios")
    print()
    if llm_wins_n == total and probe_loses_n == total:
        print("  VERDICT: both claims hold under every perturbation tested.")
        print("  The absolute rupee figures move; the ordering does not.")
    else:
        print("  VERDICT: at least one claim is FRAGILE. Scenarios marked NO above break it.")
        print("  The README must report this rather than the headline alone.")


if __name__ == "__main__":
    main()
