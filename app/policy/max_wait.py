"""Arm 4: one attempt, as late as the horizon allows. The degeneracy probe.

This arm exists to be beaten. If windows only closed at a fixed horizon, a
single attempt just before it would land inside EVERY window that exists --
gateway blips, bank outages, paydays alike -- and would beat every other arm
with zero inference. That would replace "the config decides the answer" with
"the horizon decides the answer", and a panelist asking "what if I just always
wait 13 days?" would end the demo.

`attrition_lambda` in config/world.toml is what stops that: success decays as
exp(-lambda * days_waited), so a 13.9-day attempt keeps only ~25% of its value.
Hitting a window early beats hitting it late, which is what makes precise
estimation the skill under test rather than patience.

This arm is reported in every eval from now on. IT MUST LOSE. If it ever wins,
the world model is broken and the eval should say so immediately rather than at
submission time.
"""

from __future__ import annotations

from app.models import Diagnosis, FailedPayment, PolicyDecision, RecoveryAction

#: Just inside the 14-day attrition horizon in config/world.toml.
WAIT_HOURS = 13.9 * 24


class MaxWaitPolicy:
    name = "max_wait_probe"

    def __init__(self, max_attempts: int = 1) -> None:
        self.max_attempts = max_attempts

    def decide(
        self,
        payment: FailedPayment,
        _diagnosis: Diagnosis,
        attempt_number: int,
    ) -> PolicyDecision | None:
        if attempt_number > self.max_attempts:
            return None
        return PolicyDecision(action=RecoveryAction.RETRY_AT_PAYDAY, delay_hours=WAIT_HOURS)
