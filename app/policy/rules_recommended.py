"""Arm 3: follow the rules table verbatim, including customer contact.

The contact control. Every other arm touches zero customers, which would leave
recovered-per-customer-touched as a metric only the advocated arm is penalised
by. This arm spends contacts freely -- whenever the rules table says
PROMPT_CUSTOMER_OTP / REQUEST_NEW_INSTRUMENT / REQUEST_NEW_MANDATE, it does it --
so the cost side of the ledger has a real comparison point.

It is also the strongest non-AI arm available: it gets the full action
vocabulary AND the per-reason `min_wait_hours` from the rules table, which is
hand-authored domain knowledge. If a Day-2 inference arm cannot beat this, the
inference is not earning its place.
"""

from __future__ import annotations

from app.models import Diagnosis, FailedPayment, PolicyDecision, RecoveryAction

#: Multiplier on the rules table's min_wait_hours for successive attempts.
BACKOFF_FACTOR = 2.0

#: Used when the table says wait 0h but we still need a gap between attempts,
#: e.g. re-sending a UPI collect request.
MIN_GAP_HOURS = 0.25


class RulesRecommendedPolicy:
    name = "rules_recommended"

    def __init__(self, max_attempts: int = 3) -> None:
        self.max_attempts = max_attempts

    def decide(
        self,
        payment: FailedPayment,
        diagnosis: Diagnosis,
        attempt_number: int,
    ) -> PolicyDecision | None:
        if attempt_number > self.max_attempts:
            return None

        action = diagnosis.recommended_action

        # ESCALATE and SUPPRESS make no attempt; there is no point repeating them.
        if action in (RecoveryAction.ESCALATE_MANUAL_REVIEW, RecoveryAction.SUPPRESS_DO_NOT_RETRY):
            if attempt_number == 1:
                return PolicyDecision(action, 0.0)
            return None

        # Contact actions are one-shot: asking the same customer for a new card
        # three times is harassment, not a recovery strategy.
        if action in (
            RecoveryAction.PROMPT_CUSTOMER_OTP,
            RecoveryAction.REQUEST_NEW_INSTRUMENT,
            RecoveryAction.REQUEST_NEW_MANDATE,
        ):
            if attempt_number == 1:
                return PolicyDecision(action, max(diagnosis.min_wait_hours, 0.0))
            return None

        base = max(diagnosis.min_wait_hours, MIN_GAP_HOURS)
        delay = base * (BACKOFF_FACTOR ** (attempt_number - 1))
        return PolicyDecision(action=action, delay_hours=delay)
