"""Arm 2: exponential backoff, skip failures that can never clear.

The honest control -- what a competent payments engineer builds with no AI. It
uses only the two cheapest facts in the diagnosis:

  * `retriable`  -- do not spend attempts on a card that is still expired
  * exponential backoff -- do not retry into the same outage window

It deliberately does NOT use `recommended_action` or `min_wait_hours`. Choosing
*how long* to wait for THIS payment is exactly the judgement a Day-2 classifier
is supposed to supply; handing it a per-reason wait from the rules table would
give it the intelligence under test. It gets a fixed geometric schedule, which
is what backoff actually means.

Under the window executor its reach is 21 hours. That catches bank outages
(2-3h) and issuer cooldowns, and structurally cannot reach a payday.
"""

from __future__ import annotations

from app.models import Diagnosis, FailedPayment, PolicyDecision, RecoveryAction

#: Hours before each attempt, measured from the previous one: t0+1h, +5h, +21h.
SCHEDULE_HOURS = (1.0, 4.0, 16.0)


class BackoffSkipPolicy:
    name = "backoff_skip"

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

        # A non-retriable failure will not clear on the same instrument and rail
        # no matter how long we wait. Stop immediately and spend nothing.
        if not diagnosis.retriable:
            if attempt_number == 1:
                return PolicyDecision(RecoveryAction.SUPPRESS_DO_NOT_RETRY, 0.0)
            return None

        delay = SCHEDULE_HOURS[min(attempt_number - 1, len(SCHEDULE_HOURS) - 1)]
        action = RecoveryAction.RETRY_NOW if attempt_number == 1 else RecoveryAction.RETRY_AFTER_BACKOFF
        return PolicyDecision(action=action, delay_hours=delay)
