"""The PaymentExecutor interface -- the second Day-2 swap point.

`SimulatedExecutor` implements this today. `RazorpayExecutor`, hitting test-mode
APIs, drops in behind the same call on Day 2 with no change to the orchestrator,
the policies, or the audit schema.

The interface takes a `RecoveryRequest` (payment + action + diagnosis + attempt
number) and returns an `AttemptOutcome`. Nothing about HTTP, retries-at-the-
transport-layer, credentials or idempotency keys appears here: those are an
implementation's private concern.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.models import AttemptOutcome, RecoveryRequest


@runtime_checkable
class PaymentExecutor(Protocol):
    name: str

    def execute(self, request: RecoveryRequest) -> AttemptOutcome:
        """Carry out one recovery action.

        Must not raise for an ordinary payment failure -- a failed attempt is an
        `AttemptOutcome` with status FAILED, not an exception. Exceptions are
        reserved for genuine infrastructure faults in the executor itself.
        """
        ...
