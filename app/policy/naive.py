"""Arm 1: retry everything, immediately, up to 3 times.

The strawman baseline -- what a system does with no diagnosis at all. It
receives a perfectly good `Diagnosis` and ignores every field of it. That
indifference *is* the policy, which is why the parameter is named `_diagnosis`.

Under the window executor its reach is six minutes, so it can only ever catch
failures whose window opens almost immediately (gateway blips). Everything
slower -- outages, paydays, cooldowns -- is structurally out of range.
"""

from __future__ import annotations

from app.models import Diagnosis, FailedPayment, PolicyDecision, RecoveryAction

#: Delay before each attempt, in hours since the previous one. "Immediately"
#: in production means a tight retry loop, not three calls in the same
#: microsecond, so: now, +1 min, +5 min.
SCHEDULE_HOURS = (0.0, 0.0167, 0.0833)


class NaiveRetryPolicy:
    name = "naive_retry_3x"

    def __init__(self, max_attempts: int = 3) -> None:
        self.max_attempts = max_attempts

    def decide(
        self,
        payment: FailedPayment,
        _diagnosis: Diagnosis,
        attempt_number: int,
    ) -> PolicyDecision | None:
        if attempt_number > self.max_attempts:
            return None
        delay = SCHEDULE_HOURS[min(attempt_number - 1, len(SCHEDULE_HOURS) - 1)]
        return PolicyDecision(action=RecoveryAction.RETRY_NOW, delay_hours=delay)
