"""Recovery-window executor -- outcomes decided by WHEN, not by a config number.

This replaces `SimulatedExecutor`, where P(success) was keyed on
(failure_class, action). That design made retry timing invisible to the eval,
so the one place real intelligence lives could not be measured, and any attempt
to fix it by adding timing coefficients would have meant writing the reward
function that makes my own agent win.

Here the executor knows only:

  1. the payment's hidden recovery windows (loaded from a separate file that
     no classifier or policy may import -- selfcheck L5 enforces this)
  2. six global constants from config/world.toml, none keyed by failure class
  3. when the attempt was scheduled

An attempt inside an open window succeeds with that channel's constant, decayed
for how long we waited. Outside, `p_out`. Nothing else is consulted -- in
particular the executor never reads `request.action`'s label, so a policy cannot
win by *calling* something RETRY_AT_PAYDAY. Only `scheduled_at` matters.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

from app import rng
from app.corpus.truth import CONTACT, RAIL, RETRY, WorldModel, load_ground_truth, load_world
from app.models import (
    CUSTOMER_CONTACT_ACTIONS,
    GATEWAY_ATTEMPT_ACTIONS,
    AttemptOutcome,
    AttemptStatus,
    RecoveryAction,
    RecoveryRequest,
)

#: Which channel an action exercises. SUPPRESS and ESCALATE map to nothing --
#: they make no attempt at all.
_ACTION_CHANNEL: dict[RecoveryAction, str] = {
    RecoveryAction.RETRY_NOW: RETRY,
    RecoveryAction.RETRY_AFTER_BACKOFF: RETRY,
    RecoveryAction.RETRY_AT_PAYDAY: RETRY,
    RecoveryAction.SWITCH_RAIL_TO_UPI: RAIL,
    RecoveryAction.PROMPT_CUSTOMER_OTP: CONTACT,
    RecoveryAction.REQUEST_NEW_INSTRUMENT: CONTACT,
    RecoveryAction.REQUEST_NEW_MANDATE: CONTACT,
}


class RecoveryWindowExecutor:
    """Decides outcomes by window membership plus time-decay."""

    name = "window"

    def __init__(self, ground_truth_path: Path | str, world: WorldModel | None = None) -> None:
        self._truth = load_ground_truth(ground_truth_path)
        self._world = world if world is not None else load_world()
        self._truth_path = Path(ground_truth_path)

    @property
    def model(self) -> WorldModel:
        """Named `model` so orchestrator's run-metadata hash works for both executors."""
        return self._world

    @property
    def ground_truth_path(self) -> Path:
        return self._truth_path

    def execute(self, request: RecoveryRequest) -> AttemptOutcome:
        payment = request.payment
        key = rng.stream_key(request.run_seed, payment.payment_id, request.attempt_number)

        # Actions that never touch the rails resolve without consuming a draw,
        # so suppressing in one arm cannot perturb another arm's luck.
        channel = _ACTION_CHANNEL.get(request.action)
        if channel is None:
            return AttemptOutcome(
                status=AttemptStatus.SUPPRESSED,
                amount_recovered_paise=0,
                error_reason=None,
                executor_name=self.name,
                latency_ms=0,
                rng_stream_key=key,
                success_probability=0.0,
            )

        truth = self._truth.get(payment.payment_id)
        if truth is None:
            raise KeyError(f"no ground truth for {payment.payment_id} -- corpus and truth file disagree")

        scheduled_at = datetime.fromisoformat(request.scheduled_at)
        created_at = datetime.fromisoformat(payment.created_at)
        window = truth.window(channel)

        in_window = window is not None and window[0] <= scheduled_at <= window[1]
        c = self._world.constants
        base = c.p_in(channel) if in_window else c.p_out

        # Customers leak away continuously. This is what stops "wait 13.9 days"
        # from being a winning strategy: a late attempt inside a window keeps
        # only a fraction of its value, so estimating WHEN precisely is the skill.
        days_waited = max(0.0, (scheduled_at - created_at).total_seconds() / 86400.0)
        p = base * math.exp(-c.attrition_lambda * days_waited)
        p = min(1.0, max(0.0, p))

        # The retry channel genuinely re-rolls: the blocker may have cleared and
        # conditions differ between attempts. The rail and contact channels do
        # NOT -- whether this customer has a usable alternate rail, or will
        # respond when contacted, is a fixed fact about them. Re-drawing those
        # per attempt would make three contacts beat one purely by repetition,
        # which is both physically wrong and exactly the behaviour the
        # customer-contact cost metric exists to discourage.
        if channel is RETRY:
            stream = rng.stream(request.run_seed, payment.payment_id, request.attempt_number)
        else:
            stream = rng.fixed_trait_stream(request.run_seed, payment.payment_id, channel)

        draw = stream.random()
        succeeded = draw < p
        latency_ms = int(
            rng.stream(request.run_seed, payment.payment_id, request.attempt_number).uniform(180, 2400)
        )

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
            error_reason=payment.error.reason,
            executor_name=self.name,
            latency_ms=latency_ms,
            rng_stream_key=key,
            success_probability=p,
        )


__all__ = ["RecoveryWindowExecutor", "GATEWAY_ATTEMPT_ACTIONS", "CUSTOMER_CONTACT_ACTIONS"]
