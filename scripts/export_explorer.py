"""Export a compact per-payment dataset for the interactive dashboard.

The audit trail is ~1 MB per arm, which is far too much to ship to a browser.
This flattens it to what an explorer actually needs: one row per payment, with
each arm's attempt sequence attached, using short keys to keep the payload small.

Like scripts/evaluate.py this runs AFTER every arm has finished and only
explains results, so joining the hidden windows here is legitimate — it is how
the dashboard can show *why* an attempt missed. Nothing here is visible to a
policy at decision time.

    python -m scripts.export_explorer   ->  data/runs/explorer.json
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
OUT = RUNS / "explorer.json"

_CHANNEL = {
    "RETRY_NOW": "retry", "RETRY_AFTER_BACKOFF": "retry", "RETRY_AT_PAYDAY": "retry",
    "SWITCH_RAIL_TO_UPI": "rail",
    "PROMPT_CUSTOMER_OTP": "contact", "REQUEST_NEW_INSTRUMENT": "contact",
    "REQUEST_NEW_MANDATE": "contact",
}
#: Compact action labels, so the payload does not carry the same long strings 2000 times.
_SHORT = {
    "RETRY_NOW": "retry", "RETRY_AFTER_BACKOFF": "backoff", "RETRY_AT_PAYDAY": "payday",
    "SWITCH_RAIL_TO_UPI": "rail", "PROMPT_CUSTOMER_OTP": "otp",
    "REQUEST_NEW_INSTRUMENT": "new-instrument", "REQUEST_NEW_MANDATE": "new-mandate",
    "SUPPRESS_DO_NOT_RETRY": "suppress", "ESCALATE_MANUAL_REVIEW": "escalate",
}


def main() -> None:
    manifest = json.loads((RUNS / "latest.json").read_text(encoding="utf-8"))
    payments = read_corpus(manifest["corpus"])
    truth = load_ground_truth(manifest["ground_truth"]) if manifest.get("ground_truth") else {}

    rows: dict[str, dict] = {}
    for p in payments:
        t = truth.get(p.payment_id)
        rows[p.payment_id] = {
            "id": p.payment_id,
            "rsn": p.error.reason,
            "desc": p.error.description,
            "code": p.error.code,
            "src": p.error.source,
            "step": p.error.step,
            "amt": p.amount_paise,
            "mth": str(p.method),
            "inst": p.institution,
            "at": p.created_at,
            "prior": p.prior_attempt_count,
            # The hidden truth, revealed only in this post-hoc view so the
            # dashboard can explain misses. No policy ever sees it.
            "win": {
                c: getattr(t, f"{c}_opens_at") for c in ("retry", "rail", "contact")
            } if t else {},
            "blk": t.blocker if t else "",
            "arms": {},
        }

    for arm, run_id in manifest["runs"].items():
        per: dict[str, list] = defaultdict(list)
        for r in read_jsonl(RUNS / f"{run_id}.jsonl"):
            per[r["payment_id"]].append(r)
        for pid, recs in per.items():
            recs.sort(key=lambda r: r["attempt_number"])
            row = rows.get(pid)
            if row is None:
                continue
            t = truth.get(pid)
            attempts = []
            for r in recs:
                ch = _CHANNEL.get(r["action_chosen"])
                verdict = "none"
                if ch and t:
                    w = t.window(ch)
                    if w is None:
                        verdict = "nowin"
                    else:
                        s = datetime.fromisoformat(r["scheduled_at"])
                        verdict = "in" if w[0] <= s <= w[1] else ("early" if s < w[0] else "late")
                attempts.append({
                    "n": r["attempt_number"],
                    "act": _SHORT.get(r["action_chosen"], r["action_chosen"]),
                    "d": round(r["delay_hours"], 2),
                    "at": r["scheduled_at"],
                    "out": r["outcome"],
                    "p": round(r["success_probability"], 4),
                    "v": verdict,
                })
            row["arms"][arm] = {
                "cls": recs[0]["failure_class"],
                "rule": recs[0]["rule_id"],
                "why": recs[0]["reason_text"],
                "rec": sum(r["amount_recovered_paise"] for r in recs),
                "att": attempts,
            }

    summary = json.loads((RUNS / "summary.json").read_text(encoding="utf-8"))
    payload = {
        "seed": manifest["seed"],
        "executor": manifest.get("executor", ""),
        "at_risk_paise": summary["at_risk_paise"],
        "ceiling_paise": summary["ceiling_paise"],
        "ceiling_rate": summary["ceiling_rate"],
        "ceiling_by_class": summary["ceiling_by_class"],
        "arms": [
            {k: a[k] for k in (
                "arm", "recovered_paise", "recovery_rate_value", "recovered_count",
                "n_payments", "gateway_attempts", "customers_touched",
                "in_window_attempts", "missed_early", "missed_late", "no_window_ever",
                "by_class",
            )}
            for a in summary["arms"]
        ],
        "payments": list(rows.values()),
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, {len(rows)} payments, {len(payload['arms'])} arms)")


if __name__ == "__main__":
    main()
