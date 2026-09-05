"""Arm: infer when salary lands, and put the retry there.

THE GAP THIS TARGETS
--------------------
`INSUFFICIENT_FUNDS` is the largest single class by value (Rs 93,217) and the
strongest rules arm recovers only ~52% of it. The oracle reaches 82%. The
difference is entirely timing: the rules table says "wait 72 hours", which is a
fixed guess that lands on a payday only by luck.

An account balance changes on a salary cycle. That cycle is not random — Indian
payroll clusters hard on the 1st and 7th of the month, with a smaller month-end
peak. So the right retry time is inferable from the calendar plus a prior, with
no access to anything hidden.

WHAT THIS IS AND IS NOT
-----------------------
This is RULES, not AI. It uses two things a payments engineer already knows:
the failure is a funds problem, and salaries land on predictable dates. There is
no model call and no API key. It is included precisely so the LLM arm cannot
claim credit for timing work that ordinary domain knowledge does.

It reads only `payment.created_at` and the diagnosis. It does not import
ground truth — `scripts/selfcheck.py` L5 fails the build if any policy does.
"""

from __future__ import annotations

import calendar
import math
from datetime import datetime, timedelta

from app.models import (
    Diagnosis,
    FailedPayment,
    FailureClass,
    PolicyDecision,
    RecoveryAction,
)
from app.policy.rules_recommended import RulesRecommendedPolicy

#: Prior over salary-credit day-of-month for Indian payroll. Public domain
#: knowledge, not derived from the corpus: most employers pay on the 1st, a
#: large minority on the 7th, and a tail at month-end.
PAYDAY_PRIOR: dict[int, float] = {
    1: 0.40, 2: 0.08, 5: 0.10, 7: 0.15, 10: 0.07, 15: 0.05, 25: 0.06, 28: 0.09,
}

#: Must match config/world.toml. Waiting past this recovers nothing, so a
#: candidate payday beyond it is not worth aiming at.
ATTRITION_HORIZON_DAYS = 14.0
ATTRITION_LAMBDA = 0.10


def _next_occurrence(after: datetime, dom: int) -> datetime:
    """First time day-of-month `dom` occurs strictly after `after`, at 10:00."""
    year, month = after.year, after.month
    for _ in range(3):
        day = min(dom, calendar.monthrange(year, month)[1])
        candidate = after.replace(
            year=year, month=month, day=day, hour=10, minute=0, second=0, microsecond=0
        )
        if candidate > after:
            return candidate
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return after + timedelta(days=30)


def rank_payday_delays(created_at: datetime, max_attempts: int) -> list[float]:
    """Candidate delays in hours, best expected value first.

    Scores each candidate payday by P(that is the customer's payday) x the
    attrition decay for waiting that long. That trades off "most likely date"
    against "sooner is worth more", which is the whole decision.
    """
    scored: list[tuple[float, float]] = []
    for dom, prior in PAYDAY_PRIOR.items():
        target = _next_occurrence(created_at, dom)
        days = (target - created_at).total_seconds() / 86400.0
        if days > ATTRITION_HORIZON_DAYS:
            continue  # the customer is gone before this payday arrives
        # A little past the credit, so the balance has actually posted.
        hours = days * 24.0 + 6.0
        scored.append((prior * math.exp(-ATTRITION_LAMBDA * days), hours))

    scored.sort(key=lambda x: -x[0])
    delays = [hours for _, hours in scored[:max_attempts]]

    if not delays:
        # Every plausible payday is past the horizon. Nothing to aim at, so try
        # early rather than late -- a small balance top-up can still land.
        return [24.0]

    # Convert absolute waits into gaps between successive attempts.
    delays.sort()
    gaps, prev = [], 0.0
    for d in delays:
        gaps.append(max(0.5, d - prev))
        prev = d
    return gaps


class PaydayInferencePolicy:
    """Rules-recommended everywhere, except funds failures get inferred timing."""

    name = "payday_inference"

    def __init__(self, max_attempts: int = 3) -> None:
        self.max_attempts = max_attempts
        self._base = RulesRecommendedPolicy(max_attempts=max_attempts)

    def decide(
        self,
        payment: FailedPayment,
        diagnosis: Diagnosis,
        attempt_number: int,
    ) -> PolicyDecision | None:
        if diagnosis.failure_class is not FailureClass.INSUFFICIENT_FUNDS:
            return self._base.decide(payment, diagnosis, attempt_number)

        if attempt_number > self.max_attempts:
            return None

        # A funds failure that the rules table says can never clear on this
        # instrument is not a timing problem -- defer to the base policy.
        if not diagnosis.retriable:
            return self._base.decide(payment, diagnosis, attempt_number)

        created = datetime.fromisoformat(payment.created_at)
        gaps = rank_payday_delays(created, self.max_attempts)
        if attempt_number > len(gaps):
            return None

        return PolicyDecision(
            action=RecoveryAction.RETRY_AT_PAYDAY,
            delay_hours=gaps[attempt_number - 1],
        )
