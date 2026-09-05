"""Export a compact event stream so the dashboard can replay the benchmark.

The explorer payload is ~1.1 MB because it carries every field of every attempt.
A replay needs far less: when each attempt fired, when money actually landed, and
where each payment's hidden window sat. Flattened to hours-since-corpus-start,
the whole 14 days fits in tens of kilobytes.

Post-hoc, like evaluate.py and export_explorer.py -- nothing here is visible to a
policy at decision time.

    python -m scripts.export_replay   ->  data/runs/replay.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from app.audit import read_jsonl
from app.corpus.generator import read_corpus
from app.corpus.truth import load_ground_truth

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "runs"
OUT = RUNS / "replay.json"

_CHANNEL = {
    "RETRY_NOW": "retry", "RETRY_AFTER_BACKOFF": "retry", "RETRY_AT_PAYDAY": "retry",
    "SWITCH_RAIL_TO_UPI": "rail", "PROMPT_CUSTOMER_OTP": "contact",
    "REQUEST_NEW_INSTRUMENT": "contact", "REQUEST_NEW_MANDATE": "contact",
}


def _all_advice() -> dict:
    """Run the advisor over every published Razorpay reason, once."""
    import csv as _csv

    from app.advisor import advise
    from app.diagnosis.llm_classifier import LLMClassifier
    from app.models import RazorpayError

    ref = ROOT / "data" / "reference" / "razorpay_error_reasons.csv"
    seen: dict[str, str] = {}
    with ref.open(encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            r = (row.get("Reason") or "").strip()
            if r and r not in seen:
                seen[r] = (row.get("Explanation") or "").strip()

    clf = LLMClassifier(allow_api=False, match_by_reason=True)
    out = {}
    for reason, desc in seen.items():
        err = RazorpayError(code="BAD_REQUEST_ERROR", description=desc, reason=reason,
                            source="issuer_bank", step="payment_authorization")
        out[reason] = advise(err, clf).to_dict()
    return out


def main() -> None:
    manifest = json.loads((RUNS / "latest.json").read_text(encoding="utf-8"))
    summary = json.loads((RUNS / "summary.json").read_text(encoding="utf-8"))
    payments = read_corpus(manifest["corpus"])
    truth = load_ground_truth(manifest["ground_truth"]) if manifest.get("ground_truth") else {}

    # Everything is measured in hours from the first failure in the corpus, so the
    # replay has one clock.
    t0 = min(datetime.fromisoformat(p.created_at) for p in payments)
    hours = lambda ts: (datetime.fromisoformat(ts) - t0).total_seconds() / 3600.0

    idx = {p.payment_id: i for i, p in enumerate(payments)}
    pay = []
    for p in payments:
        t = truth.get(p.payment_id)
        w = None
        if t:
            # The earliest channel that ever opens -- the moment this payment
            # became recoverable at all.
            opens = [getattr(t, f"{c}_opens_at") for c in ("retry", "rail", "contact")]
            opens = [hours(o) for o in opens if o]
            w = round(min(opens), 2) if opens else None
        pay.append({
            "r": p.error.reason,
            "a": p.amount_paise,
            "t": round(hours(p.created_at), 2),
            "w": w,                       # when a window first opens, or null = never
            "m": str(p.method),
        })

    arms = {}
    for arm, run_id in manifest["runs"].items():
        fires: list[list] = []            # [hour, payment_index, hit?]
        wins: list[list] = []             # [hour, payment_index, amount]
        for r in read_jsonl(RUNS / f"{run_id}.jsonl"):
            if not _CHANNEL.get(r["action_chosen"]):
                continue                  # suppress / escalate make no attempt
            i = idx.get(r["payment_id"])
            if i is None:
                continue
            h = round(hours(r["scheduled_at"]), 3)
            won = r["outcome"] == "succeeded"
            fires.append([h, i, 1 if won else 0])
            if won:
                wins.append([h, i, r["amount_recovered_paise"]])
        fires.sort(key=lambda x: x[0])
        wins.sort(key=lambda x: x[0])
        s = next(a for a in summary["arms"] if a["arm"] == arm)
        arms[arm] = {
            "fires": fires,
            "wins": wins,
            "total": s["recovered_paise"],
            "rate": s["recovery_rate_value"],
            "attempts": s["gateway_attempts"],
            "touched": s["customers_touched"],
            "count": s["recovered_count"],
            # Timing diagnosis: why this arm missed, not just that it did.
            "timing": {
                "in": s["in_window_attempts"], "early": s["missed_early"],
                "late": s["missed_late"], "nowin": s["no_window_ever"],
            },
            "by_class": {
                k: {"risk": v["at_risk_paise"], "rec": v["recovered_paise"],
                    "rate": v["recovery_rate_value"], "n": v["n"]}
                for k, v in s["by_class"].items()
            },
        }

    span = max(
        max((f[0] for a in arms.values() for f in a["fires"]), default=0.0),
        max((p["t"] for p in pay), default=0.0),
    )
    # --- context for the narrative pages ------------------------------------
    from app.diagnosis.classifier import RulesClassifier
    from app.diagnosis.rules import load_rule_table
    from app.models import FailureClass

    clf = RulesClassifier()
    table = load_rule_table()
    tail = [p for p in payments if clf.classify(p.error).failure_class is FailureClass.UNKNOWN]
    method_mix: dict[str, int] = defaultdict(int)
    for p in payments:
        method_mix[str(p.method)] += 1

    sens_path = RUNS / "sensitivity.json"
    sensitivity = json.loads(sens_path.read_text(encoding="utf-8")) if sens_path.exists() else None

    payload = {
        "seed": manifest["seed"],
        "rules_count": len(table),
        "codes_total": 110,
        "tail": {
            "payments": len(tail),
            "codes": len({p.error.reason for p in tail}),
            "risk": sum(p.amount_paise for p in tail),
        },
        "methods": dict(method_mix),
        "classes": {
            k: {"risk": v["at_risk_paise"], "n": v["n"],
                "ceiling": summary["ceiling_by_class"].get(k, 0)}
            for k, v in next(a for a in summary["arms"]
                             if a["arm"] == "rules_recommended")["by_class"].items()
        },
        "sensitivity": sensitivity,
        # Precomputed advice for every published reason, so the standalone build
        # of the dashboard answers offline. Same code path as /api/advise.
        "advice": _all_advice(),
        # Control defects found by diffing arms against the oracle ceiling, and
        # fixed BEFORE the LLM was measured. Each was money the AI arm would
        # otherwise have banked without inferring anything.
        "fixes": [
            {"what": "rail/contact re-rolled per attempt", "paise": 4783800,
             "kind": "executor bug", "note": "three contacts scored 0.96 instead of 0.35"},
            {"what": "card_expired recommended contact, not rail", "paise": 2627600,
             "kind": "rules-table defect", "note": "contact lands 0.35 one-shot; an alternate rail 0.65"},
            {"what": "unmapped codes escalated to nothing", "paise": 7902300,
             "kind": "strawman fallback", "note": "made the whole 36% tail score 0.0%"},
        ],
        "at_risk": summary["at_risk_paise"],
        "ceiling": summary["ceiling_paise"],
        "ceiling_rate": summary["ceiling_rate"],
        "span_h": round(span + 6, 1),
        "payments": pay,
        "arms": arms,
        "order": [a["arm"] for a in sorted(summary["arms"], key=lambda x: -x["recovered_paise"])],
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT}  ({kb:.0f} KB)")
    print(f"  {len(pay)} payments · {len(arms)} arms · span {payload['span_h']:.0f}h")
    for a, d in arms.items():
        print(f"  {a:<18} {len(d['fires']):>4} attempts, {len(d['wins']):>3} recoveries")


if __name__ == "__main__":
    main()
