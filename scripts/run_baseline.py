"""Run the control arms over the corpus.

Four arms, same corpus, same RNG streams, same hidden windows:

  naive_retry_3x    retry everything immediately, 3x        reach: 6 minutes
  backoff_skip      exponential backoff, skip non-retriable reach: 21 hours
  rules_recommended follow the rules table, contact allowed reach: ~12 days
  max_wait_probe    ONE attempt at t0+13.9d                 the degeneracy probe

`max_wait_probe` exists to be beaten. It lands inside almost every window that
exists, so if attrition decay were absent it would win with zero inference. It
is reported every run: if it ever tops the table, the world model is broken.

Usage:
    python -m scripts.run_baseline --seed 42 --n 300
    python -m scripts.run_baseline --executor legacy --corpus-version v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.corpus.generator import DEFAULT_CORPUS_DIR, corpus_path_for, read_corpus
from app.corpus.truth import ground_truth_path_for
from app.diagnosis.classifier import RulesClassifier
from app.orchestrator import run_arm
from app.policy.backoff_skip import BackoffSkipPolicy
from app.policy.max_wait import MaxWaitPolicy
from app.policy.naive import NaiveRetryPolicy
from app.policy.rules_recommended import RulesRecommendedPolicy

#: arm -> (policy class, classifier kind). `llm_recommended` runs the SAME
#: policy as `rules_recommended` and differs ONLY in the classifier, so the gap
#: between them is exactly what the model contributed -- no policy change is
#: mixed in to muddy the attribution.
ARMS: dict[str, tuple[type, str]] = {
    "naive_retry_3x": (NaiveRetryPolicy, "rules"),
    "backoff_skip": (BackoffSkipPolicy, "rules"),
    "rules_recommended": (RulesRecommendedPolicy, "rules"),
    "llm_recommended": (RulesRecommendedPolicy, "llm"),
    "max_wait_probe": (MaxWaitPolicy, "rules"),
}


def build_classifier(kind: str, allow_api: bool = False):
    """Both satisfy the same Classifier Protocol."""
    if kind == "llm":
        from app.diagnosis.llm_classifier import LLMClassifier

        return LLMClassifier(allow_api=allow_api)
    return RulesClassifier()

RUNS_DIR = Path(__file__).resolve().parents[1] / "data" / "runs"
LATEST_PATH = RUNS_DIR / "latest.json"


def build_executor(kind: str, seed: int, n: int):
    """Both executors satisfy the same PaymentExecutor Protocol."""
    if kind == "window":
        from app.execution.window import RecoveryWindowExecutor

        return RecoveryWindowExecutor(ground_truth_path_for(seed, n, DEFAULT_CORPUS_DIR))
    # Day 1's probability-keyed executor, kept so its numbers stay reproducible
    # and the switch can be shown side by side rather than just asserted.
    from app.execution.simulated import SimulatedExecutor

    return SimulatedExecutor()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run control arms.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--executor", choices=["window", "legacy"], default="window")
    ap.add_argument("--corpus-version", choices=["v1", "v2"], default="v2")
    ap.add_argument("--arm", choices=[*ARMS, "all"], default="all")
    ap.add_argument(
        "--allow-api",
        action="store_true",
        help="let the LLM arm make live API calls on a cache miss (needs ANTHROPIC_API_KEY). "
             "Off by default so the benchmark is offline and free.",
    )
    args = ap.parse_args()

    corpus_path = corpus_path_for(args.seed, args.n, version=args.corpus_version)
    if not corpus_path.exists():
        raise SystemExit(
            f"corpus not found: {corpus_path}\n"
            f"run: python -m scripts.generate_corpus --seed {args.seed} --n {args.n}"
        )

    payments = read_corpus(corpus_path)
    executor = build_executor(args.executor, args.seed, args.n)

    selected = list(ARMS) if args.arm == "all" else [args.arm]
    run_ids: dict[str, str] = {}

    print(f"corpus   {corpus_path.name}")
    print(f"executor {executor.name}")
    print()

    for arm_name in selected:
        # The probe is a single-shot arm by definition; the others honour the flag.
        cap = 1 if arm_name == "max_wait_probe" else args.max_attempts
        policy_cls, classifier_kind = ARMS[arm_name]
        policy = policy_cls(max_attempts=cap)
        classifier = build_classifier(classifier_kind, allow_api=args.allow_api)
        run_id, records = run_arm(
            payments,
            classifier=classifier,
            policy=policy,
            executor=executor,
            seed=args.seed,
            corpus_path=corpus_path,
            arm=arm_name,
        )
        run_ids[arm_name] = run_id
        recovered = sum(r.amount_recovered_paise for r in records)
        print(
            f"{arm_name:<18} run_id={run_id}  decisions={len(records):>4}"
            f"  recovered=Rs {recovered / 100:>11,.2f}"
        )
        stats = getattr(classifier, "stats", None)
        if stats:
            # Exactly where each diagnosis came from, so nobody has to take the
            # arm's headline on trust.
            print(
                f"{'':<18} classifier: rules={stats['rules_hit']} cache={stats['cache_hit']}"
                f" api={stats['api_call']} low_conf={stats['low_confidence']}"
                f" err_fallback={stats['error_fallback']} cache_miss={stats['cache_miss']}"
            )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n": args.n,
                "corpus": str(corpus_path),
                "corpus_version": args.corpus_version,
                "executor": args.executor,
                "ground_truth": (
                    str(ground_truth_path_for(args.seed, args.n, DEFAULT_CORPUS_DIR))
                    if args.executor == "window"
                    else None
                ),
                "runs": run_ids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {LATEST_PATH}")
    print("next: python -m scripts.evaluate")


if __name__ == "__main__":
    main()
