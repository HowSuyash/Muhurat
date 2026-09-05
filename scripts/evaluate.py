"""Evaluate every arm in the latest run and print the comparison.

Reports, per arm and per failure class:
  * total amount at risk
  * amount recovered and recovery rate (by money and by payment count)
  * gateway attempts spent, and how many were spent on failures that could
    never clear
  * customers touched, and recovered-per-customer-touched

That last metric is the guard against a cheap win: recovering more money by
contacting every customer is not a better system, it is a more annoying one.

Usage:
    python -m scripts.evaluate
    python -m scripts.evaluate --run-file data/runs/latest.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.audit import read_jsonl
from app.corpus.generator import read_corpus
# Loading ground truth here is legitimate and expected: evaluate.py runs AFTER
# every arm has finished, and explains results rather than making decisions.
# selfcheck L5 whitelists exactly this module, app/execution/window.py and
# selfcheck itself -- any other importer fails the build.
from app.corpus.truth import load_ground_truth, load_world

RUNS_DIR = Path(__file__).resolve().parents[1] / "data" / "runs"
LATEST_PATH = RUNS_DIR / "latest.json"

#: Which hidden channel each action exercises, for the post-hoc timing join.
_CHANNEL_OF: dict[str, str] = {
    "RETRY_NOW": "retry",
    "RETRY_AFTER_BACKOFF": "retry",
    "RETRY_AT_PAYDAY": "retry",
    "SWITCH_RAIL_TO_UPI": "rail",
    "PROMPT_CUSTOMER_OTP": "contact",
    "REQUEST_NEW_INSTRUMENT": "contact",
    "REQUEST_NEW_MANDATE": "contact",
}


def oracle_ceiling(payments, truth: dict) -> tuple[int, dict[str, int]]:
    """What a perfect window-oracle could recover, in total and per class.

    Computed here rather than in its own module so that `app/` gains no new
    importer of ground truth -- evaluate.py already runs after every arm has
    finished and is whitelisted by L5.

    The retry channel genuinely re-rolls across attempts; rail and contact are
    one-shot facts about the customer, so they get a single draw. That mirrors
    app/execution/window.py exactly.
    """
    import math
    from app.diagnosis.classifier import RulesClassifier

    c = load_world().constants
    clf = RulesClassifier()
    total = 0.0
    per_class: dict[str, float] = defaultdict(float)

    for p in payments:
        t = truth.get(p.payment_id)
        if t is None:
            continue
        created = datetime.fromisoformat(p.created_at)
        best = 0.0
        for channel in ("retry", "rail", "contact"):
            w = t.window(channel)
            if w is None:
                continue
            days = max(0.0, (w[0] - created).total_seconds() / 86400.0)
            p1 = c.p_in(channel) * math.exp(-c.attrition_lambda * days)
            tries = 3 if channel == "retry" else 1
            best = max(best, 1.0 - (1.0 - p1) ** tries)
        total += best * p.amount_paise
        per_class[str(clf.classify(p.error).failure_class)] += best * p.amount_paise

    return int(total), {k: int(v) for k, v in per_class.items()}


def rupees(paise: int) -> str:
    return f"Rs {paise / 100:>14,.2f}"


@dataclass
class ClassStats:
    n_payments: int = 0
    in_window_attempts: int = 0
    missed_early: int = 0
    missed_late: int = 0
    no_window: int = 0
    at_risk_paise: int = 0
    recovered_paise: int = 0
    recovered_count: int = 0
    gateway_attempts: int = 0
    customer_contacts: int = 0
    wasted_attempts: int = 0
    customers: set[str] = field(default_factory=set)
    touched_customers: set[str] = field(default_factory=set)


@dataclass
class ArmStats:
    arm: str
    run_id: str
    by_class: dict[str, ClassStats] = field(default_factory=lambda: defaultdict(ClassStats))
    total: ClassStats = field(default_factory=ClassStats)


def summarise(
    run_id: str,
    arm: str,
    records: list[dict],
    at_risk: dict[str, int],
    truth: dict | None = None,
) -> ArmStats:
    stats = ArmStats(arm=arm, run_id=run_id)

    # Group decisions by payment so per-payment facts are counted once.
    per_payment: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        per_payment[r["payment_id"]].append(r)

    seen_classes: dict[str, str] = {}
    for pid, rows in per_payment.items():
        rows.sort(key=lambda r: r["attempt_number"])
        fc = rows[0]["failure_class"]
        seen_classes[pid] = fc
        cs = stats.by_class[fc]

        amount = rows[0]["amount_paise"]
        cs.n_payments += 1
        cs.at_risk_paise += amount
        cs.customers.add(rows[0]["customer_id"])

        recovered = sum(r["amount_recovered_paise"] for r in rows)
        if recovered:
            cs.recovered_paise += recovered
            cs.recovered_count += 1

        succeeded = any(r["outcome"] == "succeeded" for r in rows)

        # Post-hoc timing diagnosis. This is why an arm missed: too early, too
        # late, or there was never a window at all. Joined here, never in the
        # audit trail, so no policy can see it.
        if truth is not None:
            t = truth.get(pid)
            for r in rows:
                channel = _CHANNEL_OF.get(r["action_chosen"])
                if channel is None or t is None:
                    continue
                w = t.window(channel)
                if w is None:
                    cs.no_window += 1
                    continue
                sched = datetime.fromisoformat(r["scheduled_at"])
                if w[0] <= sched <= w[1]:
                    cs.in_window_attempts += 1
                elif sched < w[0]:
                    cs.missed_early += 1
                else:
                    cs.missed_late += 1

        for r in rows:
            if r["is_gateway_attempt"]:
                cs.gateway_attempts += 1
                if not succeeded:
                    cs.wasted_attempts += 1
            if r["is_customer_contact"]:
                cs.customer_contacts += 1
                cs.touched_customers.add(r["customer_id"])

    # Roll up.
    for cs in stats.by_class.values():
        stats.total.n_payments += cs.n_payments
        stats.total.at_risk_paise += cs.at_risk_paise
        stats.total.recovered_paise += cs.recovered_paise
        stats.total.recovered_count += cs.recovered_count
        stats.total.gateway_attempts += cs.gateway_attempts
        stats.total.customer_contacts += cs.customer_contacts
        stats.total.wasted_attempts += cs.wasted_attempts
        stats.total.customers |= cs.customers
        stats.total.touched_customers |= cs.touched_customers
        stats.total.in_window_attempts += cs.in_window_attempts
        stats.total.missed_early += cs.missed_early
        stats.total.missed_late += cs.missed_late
        stats.total.no_window += cs.no_window

    # Payments the arm never attempted at all still carry money at risk.
    for pid, amount in at_risk.items():
        if pid not in per_payment:
            raise AssertionError(f"payment {pid} produced no decision record -- audit trail is incomplete")

    return stats


def print_arm(stats: ArmStats) -> None:
    t = stats.total
    money_rate = t.recovered_paise / t.at_risk_paise if t.at_risk_paise else 0.0
    count_rate = t.recovered_count / t.n_payments if t.n_payments else 0.0

    print()
    print("=" * 100)
    print(f"ARM: {stats.arm}    run_id={stats.run_id}")
    print("=" * 100)
    print(f"  Total amount at risk       {rupees(t.at_risk_paise)}   ({t.n_payments} payments)")
    print(f"  Amount recovered           {rupees(t.recovered_paise)}   ({money_rate:6.2%} of value at risk)")
    print(f"  Payments recovered         {t.recovered_count:>14} / {t.n_payments}   ({count_rate:6.2%})")
    print(f"  Gateway attempts spent     {t.gateway_attempts:>14}   ({t.wasted_attempts} on payments never recovered)")
    print(f"  Customers touched          {len(t.touched_customers):>14}   ({t.customer_contacts} contact actions)")
    if t.touched_customers:
        per_touch = t.recovered_paise / len(t.touched_customers)
        print(f"  Recovered per customer touched  {rupees(int(per_touch))}")
    else:
        print("  Recovered per customer touched             n/a   (no customer was contacted)")

    timed = t.in_window_attempts + t.missed_early + t.missed_late + t.no_window
    if timed:
        print(
            f"  Attempt timing             {t.in_window_attempts:>14} in window"
            f"   |  {t.missed_early} too early, {t.missed_late} too late,"
            f" {t.no_window} no window ever"
        )

    print()
    header = f"  {'failure class':<20} {'n':>4} {'at risk':>17} {'recovered':>17} {'rate':>7} {'attempts':>9} {'touched':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for fc in sorted(stats.by_class, key=lambda k: -stats.by_class[k].at_risk_paise):
        cs = stats.by_class[fc]
        rate = cs.recovered_paise / cs.at_risk_paise if cs.at_risk_paise else 0.0
        print(
            f"  {fc:<20} {cs.n_payments:>4} {rupees(cs.at_risk_paise)} {rupees(cs.recovered_paise)}"
            f" {rate:>6.1%} {cs.gateway_attempts:>9} {len(cs.touched_customers):>8}"
        )


def print_comparison(arms: list[ArmStats]) -> None:
    if len(arms) < 2:
        return
    print()
    print("=" * 100)
    print("HEAD TO HEAD  (same corpus, same RNG streams -- the delta is decisions, not luck)")
    print("=" * 100)
    header = (
        f"  {'arm':<18} {'recovered':>17} {'rate':>8} {'payments':>10} {'attempts':>9}"
        f" {'touched':>8} {'in-window':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    ranked = sorted(arms, key=lambda a: -a.total.recovered_paise)
    for a in ranked:
        t = a.total
        rate = t.recovered_paise / t.at_risk_paise if t.at_risk_paise else 0.0
        print(
            f"  {a.arm:<18} {rupees(t.recovered_paise)} {rate:>7.2%} {t.recovered_count:>6}/{t.n_payments:<3}"
            f" {t.gateway_attempts:>9} {len(t.touched_customers):>8} {t.in_window_attempts:>10}"
        )

    best = ranked[0]

    # The degeneracy probe must lose. If waiting until just before the attrition
    # horizon beats every arm that actually reasons, the world model is broken
    # and the eval must say so here rather than at submission time.
    probe = next((a for a in arms if a.arm == "max_wait_probe"), None)
    if probe is not None:
        print()
        if best.arm == "max_wait_probe":
            print("  *** DEGENERACY ALERT ***")
            print("  max_wait_probe (one attempt at t0+13.9d, zero inference) WON.")
            print("  Waiting beats reasoning, so the world model is broken. Do not trust these")
            print("  numbers: raise attrition_lambda in config/world.toml and re-run.")
        else:
            margin = best.total.recovered_paise - probe.total.recovered_paise
            probe_rate = (
                probe.total.recovered_paise / probe.total.at_risk_paise
                if probe.total.at_risk_paise
                else 0.0
            )
            print(
                f"  Degeneracy probe OK: max_wait_probe recovers only {probe_rate:.2%}"
                f" ({rupees(probe.total.recovered_paise).strip()}),"
            )
            print(
                f"  {rupees(margin).strip()} behind {best.arm}."
                " Waiting does not substitute for reasoning."
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate baseline arms.")
    ap.add_argument("--run-file", type=Path, default=LATEST_PATH)
    ap.add_argument("--json-out", type=Path, default=None, help="also write a machine-readable summary")
    args = ap.parse_args()

    if not args.run_file.exists():
        raise SystemExit(f"{args.run_file} not found. Run: python -m scripts.run_baseline")

    manifest = json.loads(args.run_file.read_text(encoding="utf-8"))
    payments = read_corpus(manifest["corpus"])
    at_risk = {p.payment_id: p.amount_paise for p in payments}

    truth = None
    if manifest.get("ground_truth"):
        truth = load_ground_truth(manifest["ground_truth"])

    print(f"corpus   {manifest['corpus']}")
    print(f"seed     {manifest['seed']}   payments {len(payments)}   executor {manifest.get('executor', 'legacy')}")

    arms: list[ArmStats] = []
    for arm, run_id in manifest["runs"].items():
        records = read_jsonl(RUNS_DIR / f"{run_id}.jsonl")
        stats = summarise(run_id, arm, records, at_risk, truth)
        arms.append(stats)
        print_arm(stats)

    print_comparison(arms)

    out_path = args.json_out or (RUNS_DIR / "summary.json")
    ceiling_paise, ceiling_by_class = (0, {})
    if truth is not None:
        ceiling_paise, ceiling_by_class = oracle_ceiling(payments, truth)

    total_at_risk = sum(p.amount_paise for p in payments)
    payload = {
        "seed": manifest["seed"],
        "corpus": manifest["corpus"],
        "executor": manifest.get("executor", "legacy"),
        "at_risk_paise": total_at_risk,
        "ceiling_paise": ceiling_paise,
        "ceiling_rate": (ceiling_paise / total_at_risk) if total_at_risk else 0.0,
        "ceiling_by_class": ceiling_by_class,
        "arms": [
            {
                "arm": a.arm,
                "run_id": a.run_id,
                "at_risk_paise": a.total.at_risk_paise,
                "recovered_paise": a.total.recovered_paise,
                "recovery_rate_value": (
                    a.total.recovered_paise / a.total.at_risk_paise if a.total.at_risk_paise else 0.0
                ),
                "recovered_count": a.total.recovered_count,
                "n_payments": a.total.n_payments,
                "gateway_attempts": a.total.gateway_attempts,
                "wasted_attempts": a.total.wasted_attempts,
                "customers_touched": len(a.total.touched_customers),
                "customer_contact_actions": a.total.customer_contacts,
                "in_window_attempts": a.total.in_window_attempts,
                "missed_early": a.total.missed_early,
                "missed_late": a.total.missed_late,
                "no_window_ever": a.total.no_window,
                "by_class": {
                    fc: {
                        "n": cs.n_payments,
                        "at_risk_paise": cs.at_risk_paise,
                        "recovered_paise": cs.recovered_paise,
                        "recovered_count": cs.recovered_count,
                        "recovery_rate_value": (
                            cs.recovered_paise / cs.at_risk_paise if cs.at_risk_paise else 0.0
                        ),
                        "gateway_attempts": cs.gateway_attempts,
                        "customers_touched": len(cs.touched_customers),
                    }
                    for fc, cs in a.by_class.items()
                },
            }
            for a in arms
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
