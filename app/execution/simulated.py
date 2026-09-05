"""Probabilistic outcome simulation, grounded in the diagnosed failure class.

An insufficient_funds retry does not succeed at the same rate as a gateway
timeout retry, and the model says so: every probability lives in
config/outcome_model.toml with a comment explaining it, so the assumptions are
inspectable and arguable rather than buried in code.

Determinism: outcomes are drawn from `app.rng.stream(run_seed, payment_id,
attempt_number)`, so two arms attempting the same payment at the same attempt
number see the same draw. See app/rng.py for why that matters.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

from app import rng
from app.models import (
    GATEWAY_ATTEMPT_ACTIONS,
    AttemptOutcome,
    AttemptStatus,
    FailureClass,
    RecoveryAction,
    RecoveryRequest,
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "config" / "outcome_model.toml"

#: Plausible follow-on failure reasons when a retry fails again, per class. Used
#: only to populate the audit trail with something realistic; it does not affect
#: any measured number.
_REPEAT_REASON: dict[FailureClass, str] = {
    FailureClass.INSUFFICIENT_FUNDS: "insufficient_funds",
    FailureClass.BANK_DOWNTIME: "bank_not_available",
    FailureClass.EXPIRED_CARD: "card_expired",
    FailureClass.OTP_TIMEOUT: "authentication_failed",
    FailureClass.GATEWAY_TIMEOUT: "gateway_technical_error",
    FailureClass.RISK_DECLINE: "payment_risk_check_failed",
    FailureClass.INVALID_VPA: "invalid_vpa",
    FailureClass.MANDATE_FAILURE: "mandate_creation_failed",
    FailureClass.UNKNOWN: "payment_failed",
}


@dataclass(frozen=True, slots=True)
class OutcomeModel:
    base: dict[FailureClass, dict[RecoveryAction, float]]
    attempt_decay: dict[FailureClass, float]
    prior_attempt_penalty: float
    content_sha256: str

    def probability(
        self,
        failure_class: FailureClass,
        action: RecoveryAction,
        attempt_number: int,
        prior_attempt_count: int,
    ) -> float:
        p = self.base[failure_class][action]
        p *= self.attempt_decay[failure_class] ** max(0, attempt_number - 1)
        p *= self.prior_attempt_penalty**prior_attempt_count
        # Clamp defensively; the loader already bounds the inputs, and the
        # modifiers are all in (0, 1], so this should never bind.
        return min(1.0, max(0.0, p))


def load_outcome_model(path: Path | str = DEFAULT_MODEL_PATH) -> OutcomeModel:
    """Parse and fully validate the outcome model.

    Requires a probability for every (failure_class, action) pair. A missing or
    out-of-range value fails here, loudly, rather than silently skewing results.
    """
    path = Path(path)
    raw_bytes = path.read_bytes()
    doc = tomllib.loads(raw_bytes.decode("utf-8"))

    base: dict[FailureClass, dict[RecoveryAction, float]] = {}
    for fc in FailureClass:
        section = doc["base"].get(fc.value)
        if section is None:
            raise ValueError(f"{path}: missing [base.{fc.value}] section")
        row: dict[RecoveryAction, float] = {}
        for action in RecoveryAction:
            if action.value not in section:
                raise ValueError(f"{path}: [base.{fc.value}] missing action {action.value}")
            p = float(section[action.value])
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{path}: [base.{fc.value}].{action.value} = {p} is outside [0, 1]")
            row[action] = p
        unknown = set(section) - {a.value for a in RecoveryAction}
        if unknown:
            raise ValueError(f"{path}: [base.{fc.value}] has unknown actions {sorted(unknown)}")
        base[fc] = row

    decay_section = doc["modifiers"]["attempt_decay"]
    attempt_decay: dict[FailureClass, float] = {}
    for fc in FailureClass:
        if fc.value not in decay_section:
            raise ValueError(f"{path}: [modifiers.attempt_decay] missing {fc.value}")
        d = float(decay_section[fc.value])
        if not 0.0 < d <= 1.0:
            raise ValueError(f"{path}: attempt_decay.{fc.value} = {d} must be in (0, 1]")
        attempt_decay[fc] = d

    penalty = float(doc["modifiers"]["prior_attempt_penalty"])
    if not 0.0 < penalty <= 1.0:
        raise ValueError(f"{path}: prior_attempt_penalty = {penalty} must be in (0, 1]")

    return OutcomeModel(
        base=base,
        attempt_decay=attempt_decay,
        prior_attempt_penalty=penalty,
        content_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


class SimulatedExecutor:
    """Stands in for the real Razorpay client until Day 2."""

    name = "simulated"

    def __init__(self, model: OutcomeModel | None = None) -> None:
        self._model = model if model is not None else load_outcome_model()

    @property
    def model(self) -> OutcomeModel:
        return self._model

    def execute(self, request: RecoveryRequest) -> AttemptOutcome:
        payment = request.payment
        fc = request.diagnosis.failure_class
        key = rng.stream_key(request.run_seed, payment.payment_id, request.attempt_number)

        # Actions that never touch the rails resolve without consuming a draw,
        # so suppressing a payment in one arm cannot perturb another arm's luck.
        if request.action not in GATEWAY_ATTEMPT_ACTIONS:
            return AttemptOutcome(
                status=AttemptStatus.SUPPRESSED,
                amount_recovered_paise=0,
                error_reason=None,
                executor_name=self.name,
                latency_ms=0,
                rng_stream_key=key,
                success_probability=0.0,
            )

        p = self._model.probability(
            failure_class=fc,
            action=request.action,
            attempt_number=request.attempt_number,
            prior_attempt_count=payment.prior_attempt_count,
        )

        stream = rng.stream(request.run_seed, payment.payment_id, request.attempt_number)
        draw = stream.random()
        succeeded = draw < p

        # Latency drawn from the same stream after the outcome, so it never
        # shifts the success decision. Cosmetic, for the audit trail only.
        latency_ms = int(stream.uniform(180, 2400))

        if succeeded:
            return AttemptOutcome(
                status=AttemptStatus.SUCCEEDED,
                amount_recovered_paise=payment.amount_paise,
                error_reason=None,
                executor_name=self.name,
                latency_ms=latency_ms,
                rng_stream_key=key,
                success_probability=p,
            )

        return AttemptOutcome(
            status=AttemptStatus.FAILED,
            amount_recovered_paise=0,
            error_reason=_REPEAT_REASON[fc],
            executor_name=self.name,
            latency_ms=latency_ms,
            rng_stream_key=key,
            success_probability=p,
        )
