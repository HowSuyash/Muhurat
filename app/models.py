"""Domain models for the recovery pipeline.

Deliberately stdlib dataclasses, not pydantic: the entire measurement pipeline
(corpus -> diagnosis -> execution -> eval) runs on a bare Python install. Only
the FastAPI stub and the settings module need third-party packages, so a broken
`pip install` can never block the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import StrEnum
from typing import Any


class PaymentMethod(StrEnum):
    UPI = "upi"
    CARD = "card"
    NETBANKING = "netbanking"
    EMANDATE = "emandate"


class FailureClass(StrEnum):
    """Diagnosis buckets. UNKNOWN is reachable by design -- see config/rules.toml."""

    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    BANK_DOWNTIME = "BANK_DOWNTIME"
    EXPIRED_CARD = "EXPIRED_CARD"
    OTP_TIMEOUT = "OTP_TIMEOUT"
    GATEWAY_TIMEOUT = "GATEWAY_TIMEOUT"
    RISK_DECLINE = "RISK_DECLINE"
    INVALID_VPA = "INVALID_VPA"
    MANDATE_FAILURE = "MANDATE_FAILURE"
    UNKNOWN = "UNKNOWN"


class RecoveryAction(StrEnum):
    """The bounded intervention vocabulary.

    Day 1 executes RETRY_NOW (naive arm) and RETRY_AFTER_BACKOFF /
    SUPPRESS_DO_NOT_RETRY (backoff-skip arm). The rest are declared but not yet
    selected by any policy -- Day 2 adds policies, not schema.
    """

    RETRY_NOW = "RETRY_NOW"
    RETRY_AFTER_BACKOFF = "RETRY_AFTER_BACKOFF"
    RETRY_AT_PAYDAY = "RETRY_AT_PAYDAY"
    SWITCH_RAIL_TO_UPI = "SWITCH_RAIL_TO_UPI"
    REQUEST_NEW_INSTRUMENT = "REQUEST_NEW_INSTRUMENT"
    REQUEST_NEW_MANDATE = "REQUEST_NEW_MANDATE"
    PROMPT_CUSTOMER_OTP = "PROMPT_CUSTOMER_OTP"
    SUPPRESS_DO_NOT_RETRY = "SUPPRESS_DO_NOT_RETRY"
    ESCALATE_MANUAL_REVIEW = "ESCALATE_MANUAL_REVIEW"


#: Actions that consume a customer's attention rather than a gateway attempt.
#: Recovering more money by contacting everyone is not a win, so the eval
#: reports recovered-per-customer-touched alongside the headline number.
CUSTOMER_CONTACT_ACTIONS: frozenset[RecoveryAction] = frozenset(
    {
        RecoveryAction.PROMPT_CUSTOMER_OTP,
        RecoveryAction.REQUEST_NEW_INSTRUMENT,
        RecoveryAction.REQUEST_NEW_MANDATE,
    }
)

#: Actions that put a real request on the payment rails (i.e. cost a gateway
#: attempt). SUPPRESS and ESCALATE deliberately do not.
GATEWAY_ATTEMPT_ACTIONS: frozenset[RecoveryAction] = frozenset(
    {
        RecoveryAction.RETRY_NOW,
        RecoveryAction.RETRY_AFTER_BACKOFF,
        RecoveryAction.RETRY_AT_PAYDAY,
        RecoveryAction.SWITCH_RAIL_TO_UPI,
    }
)


class AttemptStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SUPPRESSED = "suppressed"  # no attempt made; the policy chose to stop


@dataclass(frozen=True, slots=True)
class RazorpayError:
    """The full error object as Razorpay returns it.

    The Classifier receives this whole object rather than a code string, so a
    Day-2 LLM implementation has `description` prose, `source` and `step` to
    reason over without any interface change.
    """

    code: str  # BAD_REQUEST_ERROR | GATEWAY_ERROR | SERVER_ERROR
    description: str  # human-readable gateway message
    reason: str  # the 114-value enum; the rules table's match key
    source: str  # customer | bank | gateway | issuer_bank | internal | ...
    step: str  # payment_initiation | payment_authentication | ...
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PriorAttempt:
    """A failure that already happened before this record reached the pipeline."""

    ts: str  # ISO-8601
    reason: str
    status: str


@dataclass(frozen=True, slots=True)
class FailedPayment:
    """The agent-visible record. Everything a Classifier or Policy may know.

    There is deliberately no recovery-window field here, and no place to hide
    one: the hidden truth lives in a separate file loaded only by the executor
    (app/corpus/truth.py). scripts/selfcheck.py asserts that structurally.
    """

    payment_id: str
    customer_id: str
    amount_paise: int  # Razorpay's native unit; never store rupees as float
    currency: str
    method: PaymentMethod
    created_at: str  # ISO-8601, when the failure occurred
    error: RazorpayError
    # Which bank or PSP handled this payment. Legitimate, observable signal:
    # correlated failures across the same institution in a time neighbourhood
    # are how downtime is actually detected. Inferring an outage from these is
    # the skill under test, not a leak.
    institution: str = ""
    prior_attempts: tuple[PriorAttempt, ...] = ()

    @property
    def prior_attempt_count(self) -> int:
        return len(self.prior_attempts)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["method"] = str(self.method)
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "FailedPayment":
        return FailedPayment(
            payment_id=d["payment_id"],
            customer_id=d["customer_id"],
            amount_paise=d["amount_paise"],
            currency=d["currency"],
            method=PaymentMethod(d["method"]),
            created_at=d["created_at"],
            error=RazorpayError(**d["error"]),
            institution=d.get("institution", ""),
            prior_attempts=tuple(PriorAttempt(**p) for p in d.get("prior_attempts", [])),
        )


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """What a Classifier returns. Provider-agnostic by construction."""

    failure_class: FailureClass
    recommended_action: RecoveryAction
    retriable: bool
    min_wait_hours: float
    confidence: float
    reason: str  # human-readable justification, for the audit trail
    classifier_name: str
    classifier_version: str
    rule_id: str | None = None  # None for a non-rules classifier


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """What a policy chose, and -- crucially -- when.

    `action` is declared *intent*: it is recorded in the audit trail and never
    read by the executor. `delay_hours` is the actual bet. A policy cannot win
    by labelling something RETRY_AT_PAYDAY; it has to estimate when payday is
    and put that number here.
    """

    action: RecoveryAction
    delay_hours: float  # since the previous attempt (or the failure, for attempt 1)


@dataclass(frozen=True, slots=True)
class RecoveryRequest:
    """One intervention to execute against a PaymentExecutor."""

    payment: FailedPayment
    action: RecoveryAction
    diagnosis: Diagnosis
    attempt_number: int  # 1-based
    run_seed: int
    scheduled_at: str = ""  # ISO-8601; when this attempt actually goes out
    delay_hours: float = 0.0  # since the previous attempt, for the audit trail


@dataclass(frozen=True, slots=True)
class AttemptOutcome:
    status: AttemptStatus
    amount_recovered_paise: int
    error_reason: str | None  # populated when status is FAILED
    executor_name: str
    latency_ms: int
    rng_stream_key: str  # exact key used, so any outcome can be reproduced
    success_probability: float  # the p actually used, for audit


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """One row per attempt. The audit trail is a scoring criterion, so this
    schema exists from Day 1 rather than being retrofitted.

    Captures: input -> rule fired -> reason -> action chosen -> outcome.
    """

    decision_id: str
    run_id: str
    ts: str
    arm: str

    # --- input ---
    payment_id: str
    customer_id: str
    amount_paise: int
    method: str
    error_code: str
    error_reason: str
    gateway_message: str
    error_source: str
    error_step: str
    institution: str
    prior_attempt_count: int
    attempt_number: int
    # When this attempt went out, and how long we waited for it. These are the
    # decision -- but note the audit deliberately does NOT record whether the
    # attempt landed inside a window. evaluate.py joins the truth at report time
    # instead, so the audit trail carries zero leak surface.
    scheduled_at: str
    delay_hours: float

    # --- rule fired ---
    classifier_name: str
    classifier_version: str
    rule_id: str | None
    confidence: float

    # --- reason ---
    failure_class: str
    reason_text: str

    # --- action chosen ---
    policy_name: str
    action_chosen: str
    recommended_action: str  # what diagnosis suggested, which a policy may ignore
    is_customer_contact: bool
    is_gateway_attempt: bool

    # --- outcome ---
    executor_name: str
    outcome: str
    outcome_error_reason: str | None
    amount_recovered_paise: int
    success_probability: float
    latency_ms: int
    rng_stream_key: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Column order for the SQLite `decisions` table, derived from the dataclass so
#: the two can never drift apart.
DECISION_FIELDS: tuple[str, ...] = tuple(DecisionRecord.__dataclass_fields__)
