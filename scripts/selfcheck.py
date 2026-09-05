"""Assertions that make the reported numbers trustworthy.

Not a test suite -- a set of checks that fail loudly if any claim in the README
stops being true. Run it after any change to the configs or the pipeline.

Usage:
    python -m scripts.selfcheck
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from app.audit import read_jsonl
from app.corpus.generator import (
    DEFAULT_CORPUS_DIR,
    corpus_path_for,
    generate_corpus,
    load_reason_descriptions,
    read_corpus,
)
from app.corpus.truth import ground_truth_path_for, load_ground_truth, load_world
from scripts.leakcheck import run_leak_checks
from app.diagnosis.classifier import RulesClassifier
from app.diagnosis.rules import load_rule_table
from app.execution.simulated import load_outcome_model
from app.models import FailureClass, RecoveryAction, RecoveryRequest

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "data" / "runs"
LATEST = RUNS_DIR / "latest.json"

_failures: list[str] = []
_passed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _passed
    if condition:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failures.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def main() -> None:
    print("selfcheck")
    print("-" * 78)

    table = load_rule_table()
    model = load_outcome_model()

    # 1. Every probability is a probability.
    bad = [
        (fc, a, p)
        for fc, row in model.base.items()
        for a, p in row.items()
        if not 0.0 <= p <= 1.0
    ]
    check("all success probabilities in [0, 1]", not bad, str(bad[:3]))

    # 2. The outcome model covers every (class, action) pair -- no silent gaps.
    missing = [
        (fc.value, a.value)
        for fc in FailureClass
        for a in RecoveryAction
        if a not in model.base.get(fc, {})
    ]
    check("outcome model covers every (class, action) pair", not missing, str(missing[:3]))

    # 2b. THE INTEGRITY CLAIM: world.toml contains no failure-class name. If a
    #     class name appeared here, a probability could be tuned per class and
    #     the reward function would once again be something I wrote.
    world_text = (ROOT / "config" / "world.toml").read_text(encoding="utf-8")
    class_names = [fc.value for fc in FailureClass if fc.value in world_text]
    check(
        "world.toml is not keyed by failure class (the integrity claim)",
        not class_names,
        f"found {class_names}",
    )

    world = load_world()
    check(
        "attrition_lambda > 0 (otherwise waiting until the horizon dominates)",
        world.constants.attrition_lambda > 0,
        f"lambda = {world.constants.attrition_lambda}",
    )

    # 3. Corpus generation is reproducible.
    a, a_truth = generate_corpus(n=300, seed=42)
    b, _ = generate_corpus(n=300, seed=42)
    ha = hashlib.sha256(json.dumps([p.to_dict() for p in a], sort_keys=True).encode()).hexdigest()
    hb = hashlib.sha256(json.dumps([p.to_dict() for p in b], sort_keys=True).encode()).hexdigest()
    check("corpus is reproducible for a fixed seed", ha == hb, f"{ha[:12]} != {hb[:12]}")

    # 4. A different seed gives a different corpus (the seed actually does something).
    c, _ = generate_corpus(n=300, seed=43)
    hc = hashlib.sha256(json.dumps([p.to_dict() for p in c], sort_keys=True).encode()).hexdigest()
    check("a different seed gives a different corpus", ha != hc)

    # 4b. The hidden truth is reproducible too, and every payment has one.
    ha_t = hashlib.sha256(json.dumps([t.to_dict() for t in a_truth], sort_keys=True).encode()).hexdigest()
    _, b_truth = generate_corpus(n=300, seed=42)
    hb_t = hashlib.sha256(json.dumps([t.to_dict() for t in b_truth], sort_keys=True).encode()).hexdigest()
    check("ground truth is reproducible for a fixed seed", ha_t == hb_t)
    check(
        "every payment has exactly one truth record",
        {t.payment_id for t in a_truth} == {p.payment_id for p in a},
    )

    # 5. Every corpus reason is a REAL published Razorpay reason, and every one
    #    has recovery semantics defined. Unmapped is expected; unknown-to-the-
    #    world-model is a bug.
    classifier = RulesClassifier(table)
    official = set(load_reason_descriptions())
    invented = {p.error.reason for p in a if p.error.reason not in official}
    check("every corpus reason is a real published Razorpay reason", not invented, str(sorted(invented)[:5]))

    undefined = {p.error.reason for p in a if world.spec_for(p.error.reason) is None}
    check("every corpus reason has recovery semantics in world.toml", not undefined, str(sorted(undefined)[:5]))

    # 6. The long tail is genuinely large. Razorpay publishes 110 reasons and a
    #    hand-written table covers only the head, so a corpus that under-samples
    #    the tail flatters every rules-based arm. Guard the design intent.
    unknown_n = sum(1 for p in a if classifier.classify(p.error).failure_class is FailureClass.UNKNOWN)
    share = unknown_n / len(a)
    check(
        "unmapped long tail is 25-45% of the corpus",
        0.25 <= share <= 0.45,
        f"{unknown_n}/{len(a)} = {share:.1%}",
    )

    # 6b. The tail must be genuinely varied -- not three codes repeated.
    tail_codes = {
        p.error.reason for p in a
        if classifier.classify(p.error).failure_class is FailureClass.UNKNOWN
    }
    check("the tail spans many distinct codes", len(tail_codes) >= 25, f"only {len(tail_codes)} distinct")

    # 6c. The tail must be behaviourally MIXED. If every tail code were terminal,
    #     "escalate everything unmapped" would be optimal and generalising would
    #     buy nothing -- the gap has to be worth closing.
    tail_retriable = sum(
        1 for r in tail_codes if (world.spec_for(r) or {}).get("retry") != "never"
    )
    check(
        "the tail is behaviourally mixed, not uniformly terminal",
        0.15 <= tail_retriable / max(1, len(tail_codes)) <= 0.85,
        f"{tail_retriable}/{len(tail_codes)} have a retry window",
    )

    # 7. Every corpus record has a real gateway message from Razorpay's own list.
    empty = [p.payment_id for p in a if not p.error.description.strip()]
    check("every record carries a real Razorpay gateway message", not empty, f"{len(empty)} empty")

    # --- checks that need a completed run -------------------------------------
    if not LATEST.exists():
        print("\n  SKIP  run-dependent checks (no data/runs/latest.json)")
        print("        run: python -m scripts.run_baseline")
    else:
        manifest = json.loads(LATEST.read_text(encoding="utf-8"))
        payments = read_corpus(manifest["corpus"])
        at_risk_total = sum(p.amount_paise for p in payments)

        for arm, run_id in manifest["runs"].items():
            records = read_jsonl(RUNS_DIR / f"{run_id}.jsonl")

            # 8. Every payment produced at least one decision record.
            seen = {r["payment_id"] for r in records}
            check(
                f"[{arm}] every payment has an audit record",
                seen == {p.payment_id for p in payments},
                f"{len(seen)} of {len(payments)}",
            )

            # 9. Attempt cap is respected (the probe is single-shot by design).
            cap = 1 if arm == "max_wait_probe" else 3
            over = [r["payment_id"] for r in records if r["attempt_number"] > cap]
            check(f"[{arm}] no payment exceeds {cap} attempt(s)", not over, str(over[:3]))

            # 9b. Attempts move forward in time and never precede the failure.
            bad_clock = []
            for pid, rows in sorted(
                {r["payment_id"]: None for r in records}.items()
            ):
                pass
            seq: dict[str, list] = defaultdict(list)
            for r in records:
                seq[r["payment_id"]].append(r)
            for pid, rows in seq.items():
                rows.sort(key=lambda r: r["attempt_number"])
                prev = None
                for r in rows:
                    if r["delay_hours"] < 0:
                        bad_clock.append(pid)
                    if prev is not None and r["scheduled_at"] < prev:
                        bad_clock.append(pid)
                    prev = r["scheduled_at"]
            check(f"[{arm}] attempt timestamps advance monotonically", not bad_clock, str(bad_clock[:3]))

            # 10. Nothing is attempted after it already succeeded.
            by_payment: dict[str, list[dict]] = defaultdict(list)
            for r in records:
                by_payment[r["payment_id"]].append(r)
            leaked = []
            for pid, rows in by_payment.items():
                rows.sort(key=lambda r: r["attempt_number"])
                for i, r in enumerate(rows):
                    if r["outcome"] == "succeeded" and i != len(rows) - 1:
                        leaked.append(pid)
            check(f"[{arm}] no attempts after a success", not leaked, str(leaked[:3]))

            # 11. Recovered never exceeds at risk, and per-class sums reconcile.
            recovered = sum(r["amount_recovered_paise"] for r in records)
            check(
                f"[{arm}] recovered <= at risk",
                recovered <= at_risk_total,
                f"{recovered} > {at_risk_total}",
            )

            per_class: dict[str, int] = defaultdict(int)
            for r in records:
                per_class[r["failure_class"]] += r["amount_recovered_paise"]
            check(
                f"[{arm}] per-class recovered sums to total",
                sum(per_class.values()) == recovered,
                f"{sum(per_class.values())} != {recovered}",
            )

            # 12. A recovered payment is recovered exactly once.
            double = [
                pid
                for pid, rows in by_payment.items()
                if sum(1 for r in rows if r["outcome"] == "succeeded") > 1
            ]
            check(f"[{arm}] no payment recovered twice", not double, str(double[:3]))

            # 13. Every audit row is complete -- the trail is a scoring criterion.
            required = (
                "error_reason", "failure_class", "reason_text",
                "action_chosen", "outcome", "rng_stream_key",
            )
            incomplete = [
                r["decision_id"] for r in records
                if any(r.get(f) in (None, "") for f in required)
            ]
            check(f"[{arm}] every audit row is fully populated", not incomplete, str(incomplete[:3]))

        # 14. Common random numbers: both arms must draw from identical streams
        #     for the same (payment, attempt). This is what makes the comparison
        #     attributable to decisions rather than luck.
        arm_names = list(manifest["runs"])
        if len(arm_names) >= 2:
            streams = []
            for arm in arm_names[:2]:
                rows = read_jsonl(RUNS_DIR / f"{manifest['runs'][arm]}.jsonl")
                streams.append({(r["payment_id"], r["attempt_number"]): r["rng_stream_key"] for r in rows})
            shared = set(streams[0]) & set(streams[1])
            mismatched = [k for k in shared if streams[0][k] != streams[1][k]]
            check(
                "common random numbers: shared (payment, attempt) cells use identical streams",
                not mismatched,
                f"{len(mismatched)} mismatches of {len(shared)} shared cells",
            )

    # --- repetition must not manufacture recovery -----------------------------
    # Whether a customer has a usable alternate rail, or will answer when
    # contacted, is a fixed fact about them. If those re-rolled per attempt,
    # three contacts would beat one at 0.96 vs 0.35 and an arm could farm
    # recovery by pestering people -- the exact behaviour the contact-cost
    # metric exists to discourage.
    truth_p = ground_truth_path_for(42, 300, DEFAULT_CORPUS_DIR)
    if truth_p.exists():
        from app.execution.window import RecoveryWindowExecutor

        ex = RecoveryWindowExecutor(truth_p)
        sample = read_corpus(corpus_path_for(42, 300))[:80]
        divergent = []
        for pay in sample:
            for action in (
                RecoveryAction.REQUEST_NEW_INSTRUMENT,
                RecoveryAction.SWITCH_RAIL_TO_UPI,
            ):
                d = classifier.classify(pay.error)
                outs = {
                    ex.execute(
                        RecoveryRequest(
                            payment=pay,
                            action=action,
                            diagnosis=d,
                            attempt_number=k,
                            run_seed=42,
                            scheduled_at=pay.created_at,
                            delay_hours=0.0,
                        )
                    ).status
                    for k in (1, 2, 3)
                }
                if len(outs) > 1:
                    divergent.append((pay.payment_id, str(action)))
        check(
            "repeating a rail/contact action cannot change its outcome",
            not divergent,
            str(divergent[:3]),
        )

    # --- L1-L8: the integrity claim -------------------------------------------
    truth_path = ground_truth_path_for(42, 300, DEFAULT_CORPUS_DIR)
    corpus_file = corpus_path_for(42, 300)
    if truth_path.exists() and corpus_file.exists():
        print()
        print("  leak prevention (L1-L8) -- can any arm read the hidden windows?")
        truth_map = load_ground_truth(truth_path)
        visible = read_corpus(corpus_file)
        diagnoses = [classifier.classify(p.error) for p in visible]
        run_leak_checks(check, corpus_file, truth_map, visible, diagnoses)
    else:
        print()
        print("  SKIP  L1-L8 leak checks (generate the v2 corpus first)")

    # --- the degeneracy probe must lose ---------------------------------------
    summary_path = RUNS_DIR / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        arms = {a["arm"]: a for a in summary["arms"]}
        probe = arms.get("max_wait_probe")
        if probe is not None:
            best = max(summary["arms"], key=lambda a: a["recovered_paise"])
            check(
                "max_wait_probe LOSES (waiting is not a substitute for reasoning)",
                best["arm"] != "max_wait_probe",
                f"probe won with {probe['recovered_paise']}",
            )
            # It should also land in plenty of windows -- that is what makes it a
            # meaningful probe rather than an arm that simply never tries.
            check(
                "max_wait_probe does hit windows (so its loss is due to lateness)",
                probe["in_window_attempts"] > 0.4 * probe["n_payments"],
                f"only {probe['in_window_attempts']} in-window of {probe['n_payments']}",
            )

    print("-" * 78)
    if _failures:
        print(f"{_passed} passed, {len(_failures)} FAILED")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print(f"{_passed} passed, 0 failed")


if __name__ == "__main__":
    main()
