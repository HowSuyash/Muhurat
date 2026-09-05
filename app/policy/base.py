"""The RecoveryPolicy interface.

A policy turns a diagnosis into an action AND a delay. Keeping this separate
from the Classifier matters: Day 2 changes *how well we diagnose* (a new
Classifier) independently of *what we do about it* (a new Policy), and the eval
can attribute the gain to one or the other.

Day-2 change: `decide` now returns a `PolicyDecision` carrying `delay_hours`.
Timing cannot be the decision variable unless a policy can express it, and the
executor reads only the delay -- never the action label.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.models import Diagnosis, FailedPayment, PolicyDecision


@runtime_checkable
class RecoveryPolicy(Protocol):
    name: str
    max_attempts: int

    def decide(
        self,
        payment: FailedPayment,
        diagnosis: Diagnosis,
        attempt_number: int,
    ) -> PolicyDecision | None:
        """Choose the action and delay for this attempt, or None to stop.

        `attempt_number` is 1-based and counts attempts within this run only;
        failures that predate ingest live on `payment.prior_attempts`.
        `delay_hours` is measured from the previous attempt, or from the
        original failure for attempt 1.
        """
        ...
