"""The recovery advisor: the winning policy, exposed as a tool.

A merchant pastes in a failed payment and gets back a decision — what went
wrong, whether it can be recovered at all, **when** to try, and what to say to
the customer.

WHAT IT IS ALLOWED TO KNOW
--------------------------
Exactly what a policy sees during the benchmark: the Razorpay error object, the
hand-written rules table, and the LLM classifier for codes the table cannot
place. It deliberately does NOT read config/world.toml or the ground truth, even
though both sit in this repo and would make its answers look better.

That restraint is the point. If the advisor consulted the hidden windows it
would be an oracle, not a product, and its advice would be untestable. Because
it uses only what `llm_recommended` uses, its advice IS that arm's behaviour —
so the 58.11% measured in the benchmark is the number this tool actually earns.

`scripts/selfcheck.py` L5/L5a enforce that: nothing under `app/` may import the
ground truth except the executor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.diagnosis.classifier import RulesClassifier
from app.models import Diagnosis, PaymentMethod, RazorpayError, RecoveryAction

#: Per-action playbook. Written for the two people who actually act on it: the
#: merchant deciding what their system should do, and the customer receiving a
#: message. Kept as data so the copy is reviewable without reading code.
PLAYBOOK: dict[RecoveryAction, dict[str, Any]] = {
    RecoveryAction.RETRY_NOW: {
        "verdict": "Retry almost immediately",
        "why": "This is a transient fault on the rails, not a decision about the customer. "
               "Nothing about the payment itself was refused.",
        "merchant": [
            "Re-submit the same payment within a few minutes.",
            "Use the same idempotency key so a late success upstream cannot double-charge.",
            "Allow up to three attempts; if all fail, treat it as a real decline and re-diagnose.",
        ],
        "avoid": ["Do not ask the customer for anything — they did nothing wrong and may not even know it failed."],
        "customer_subject": "We're completing your payment",
        "customer_body": "A brief technical issue interrupted your payment. We're retrying it "
                         "automatically — there's nothing you need to do.",
        "contacts_customer": False,
    },
    RecoveryAction.RETRY_AFTER_BACKOFF: {
        "verdict": "Wait, then retry",
        "why": "The blocker is real but temporary — an outage, a cutoff window, or a limit that "
               "resets. Retrying before it clears just spends an attempt inside the same failure.",
        "merchant": [
            "Schedule the retry for the time below rather than firing immediately.",
            "If many payments on this bank are failing together, wait for the cluster to stop before retrying any.",
            "Space further attempts out; do not compress them into the first hour.",
        ],
        "avoid": ["Do not retry in a tight loop — it adds load to the thing that is already failing."],
        "customer_subject": "We'll retry your payment shortly",
        "customer_body": "Your bank is briefly unavailable. We'll retry your payment automatically "
                         "once it's back — no action needed from you.",
        "contacts_customer": False,
    },
    RecoveryAction.RETRY_AT_PAYDAY: {
        "verdict": "Wait for the salary cycle",
        "why": "The account is empty. A balance refills on a salary or statement cycle — days away, "
               "not seconds. Timing is the entire intervention here.",
        "merchant": [
            "Schedule the retry near the customer's likely credit date, not on a fixed timer.",
            "If you know this customer's past successful payment dates, align to that day of the month.",
            "One well-timed attempt beats three early ones and costs a third as much.",
        ],
        "avoid": [
            "Do not retry within minutes — the balance will not have moved.",
            "Do not keep retrying daily; each failure can harden the issuer's response.",
        ],
        "customer_subject": "Your payment didn't go through",
        "customer_body": "Your payment couldn't be completed because the account didn't have "
                         "sufficient funds. We'll try again automatically in a few days, or you can "
                         "pay now with a different method.",
        "contacts_customer": False,
    },
    RecoveryAction.SWITCH_RAIL_TO_UPI: {
        "verdict": "Move it to a different rail",
        "why": "This instrument will not clear — but the customer can still pay. Changing the rail "
               "sidesteps the instrument-specific problem without asking them to fix anything.",
        "merchant": [
            "Offer UPI as the primary option on the retry screen, not as a secondary link.",
            "Carry the amount and order across so the customer does not re-enter anything.",
            "Treat this as one attempt: whether an alternate rail is available is a fixed fact about "
            "the customer, so offering it three times is worth no more than offering it once.",
        ],
        "avoid": ["Do not retry the original card — it will refuse again for the same reason."],
        "customer_subject": "Try a different payment method",
        "customer_body": "Your card couldn't complete this payment. Paying by UPI takes a few "
                         "seconds and usually goes through straight away.",
        "contacts_customer": False,
    },
    RecoveryAction.REQUEST_NEW_INSTRUMENT: {
        "verdict": "Ask the customer for a different instrument",
        "why": "The stored instrument is dead or wrong, and no amount of waiting changes that. "
               "Only the customer can supply a working one.",
        "merchant": [
            "Send one clear message asking for an updated card or UPI ID.",
            "Deep-link straight to the update screen — every extra step loses people.",
            "Ask once. A second and third ask recover almost nothing and cost goodwill.",
        ],
        "avoid": [
            "Do not retry the stored instrument in the background; it cannot succeed.",
            "Do not chase the same customer repeatedly.",
        ],
        "customer_subject": "Update your payment details",
        "customer_body": "We couldn't process your payment with the details we have on file. "
                         "Adding a current card or UPI ID takes under a minute.",
        "contacts_customer": True,
    },
    RecoveryAction.REQUEST_NEW_MANDATE: {
        "verdict": "Re-register the mandate",
        "why": "The mandate itself is in a terminal state. Subsequent charges cannot run against it, "
               "so it has to be created again.",
        "merchant": [
            "Start a fresh mandate registration rather than retrying the charge.",
            "Explain that this authorises future payments, so the customer knows why they are asked twice.",
            "Pause dependent billing until the new mandate is confirmed.",
        ],
        "avoid": ["Do not retry the charge against the dead mandate — every attempt fails identically."],
        "customer_subject": "Re-authorise your automatic payments",
        "customer_body": "The standing authorisation for your payments has lapsed. Re-approving it "
                         "keeps your subscription running without interruption.",
        "contacts_customer": True,
    },
    RecoveryAction.PROMPT_CUSTOMER_OTP: {
        "verdict": "Re-engage the customer to authenticate",
        "why": "The instrument is fine — authentication did not complete. A silent retry cannot "
               "produce an OTP the customer never entered.",
        "merchant": [
            "Send a fresh payment link so a new OTP is issued.",
            "Send it while intent is still warm — within the hour, not the next day.",
            "Consider a lower-friction rail: UPI PIN entry completes far more often than card 3DS.",
        ],
        "avoid": ["Do not retry silently in the background — there is no one there to authenticate."],
        "customer_subject": "Finish your payment",
        "customer_body": "Your payment stopped at the verification step. Here's a fresh link — "
                         "it takes a few seconds to complete.",
        "contacts_customer": True,
    },
    RecoveryAction.SUPPRESS_DO_NOT_RETRY: {
        "verdict": "Stop — do not retry",
        "why": "There is nothing to recover. Either the payment already succeeded, or the request "
               "itself is the problem. Another attempt would be wrong, not merely useless.",
        "merchant": [
            "Check whether an earlier attempt on this order already succeeded before doing anything.",
            "Fix the request or the configuration that produced this, then let the customer retry normally.",
        ],
        "avoid": [
            "Do not retry — you risk a duplicate charge.",
            "Do not message the customer; this is not their problem to solve.",
        ],
        "customer_subject": None,
        "customer_body": None,
        "contacts_customer": False,
    },
    RecoveryAction.ESCALATE_MANUAL_REVIEW: {
        "verdict": "Escalate to a human",
        "why": "This carries a risk or compliance signal. Automated retries are inappropriate here "
               "and repeated attempts can make the underlying flag worse.",
        "merchant": [
            "Route to your risk queue with the full error object attached.",
            "Hold further automated attempts on this customer until it is reviewed.",
        ],
        "avoid": [
            "Do not retry — repeat attempts raise the risk score.",
            "Do not tell the customer it was a risk decline.",
        ],
        "customer_subject": "We couldn't complete your payment",
        "customer_body": "We weren't able to process this payment. Please try a different method, "
                         "or contact your bank if it keeps happening.",
        "contacts_customer": False,
    },
}


@dataclass(frozen=True, slots=True)
class Advice:
    reason: str
    failure_class: str
    retriable: bool
    confidence: float
    decided_by: str            # "rules table" or "LLM (unmapped code)"
    rule_id: str | None
    action: str
    verdict: str
    why: str
    wait_hours: float
    retry_at: str | None       # ISO-8601, or None when there is nothing to schedule
    retry_in_words: str
    contacts_customer: bool
    merchant_steps: list[str]
    avoid: list[str]
    customer_subject: str | None
    customer_message: str | None
    diagnosis_note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _in_words(h: float) -> str:
    if h <= 0:
        return "immediately"
    if h < 1:
        return f"in about {int(round(h * 60))} minutes"
    if h < 48:
        return f"in about {h:g} hours"
    return f"in about {h / 24:.0f} days"


def advise(
    error: RazorpayError,
    classifier: Any,
    *,
    failed_at: datetime | None = None,
    amount_paise: int | None = None,
    method: PaymentMethod | str | None = None,
    prior_attempts: int = 0,
) -> Advice:
    """Diagnose one failure and say what to do about it, and when."""
    diagnosis: Diagnosis = classifier.classify(error)
    action = diagnosis.recommended_action
    play = PLAYBOOK[action]

    wait = float(diagnosis.min_wait_hours)
    # Each failure already on the record is evidence the easy window has passed,
    # so back off further rather than repeating the same schedule.
    if prior_attempts and wait > 0:
        wait *= 1.6 ** min(prior_attempts, 3)

    schedulable = action not in (
        RecoveryAction.SUPPRESS_DO_NOT_RETRY,
        RecoveryAction.ESCALATE_MANUAL_REVIEW,
    )
    base = failed_at or datetime.now(timezone.utc)
    retry_at = (base + timedelta(hours=wait)).isoformat() if schedulable else None

    decided_by = "rules table" if diagnosis.rule_id else "LLM (code not in the rules table)"
    if diagnosis.rule_id is None and "fallback" in diagnosis.reason.lower():
        decided_by = "rules fallback (LLM unavailable or unsure)"

    return Advice(
        reason=error.reason,
        failure_class=str(diagnosis.failure_class),
        retriable=diagnosis.retriable,
        confidence=diagnosis.confidence,
        decided_by=decided_by,
        rule_id=diagnosis.rule_id,
        action=str(action),
        verdict=play["verdict"],
        why=play["why"],
        wait_hours=round(wait, 2),
        retry_at=retry_at,
        retry_in_words=_in_words(wait) if schedulable else "not scheduled",
        contacts_customer=bool(play["contacts_customer"]),
        merchant_steps=list(play["merchant"]),
        avoid=list(play["avoid"]),
        customer_subject=play["customer_subject"],
        customer_message=play["customer_body"],
        diagnosis_note=diagnosis.reason,
    )


def known_reasons(classifier: Any = None) -> list[str]:
    """Every reason the rules table covers, for the advisor's picker."""
    clf = classifier or RulesClassifier()
    return sorted(clf.table.covered_reasons)
