"""Hidden recovery windows -- the ground truth an agent must infer, never read.

WHY THIS FILE IS SEPARATE
-------------------------
The truth is written to its own JSONL file and loaded only by the executor. It
is never a field on `FailedPayment`, never nested in `error.metadata`, never
reachable from any object a Classifier or Policy is handed. That is a structural
guarantee rather than a naming convention, and `scripts/selfcheck.py` asserts it
eight different ways (L1-L8), including an AST scan that fails the build if any
module outside the executor imports `load_ground_truth`.

WHAT A WINDOW MEANS
-------------------
Each payment gets up to three recovery channels, each with an opening time
derived from the failure's own semantics (config/world.toml):

    retry   -- same instrument, same rail, later
    rail    -- move to a different rail
    contact -- ask the customer to act

An attempt inside an open window succeeds with that channel's global constant,
decayed by how long we waited. Outside, `p_out`. There is no per-failure-class
probability anywhere: all class differentiation comes from WHEN the window opens.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app import rng

DEFAULT_WORLD_PATH = Path(__file__).resolve().parents[2] / "config" / "world.toml"

RETRY = "retry"
RAIL = "rail"
CONTACT = "contact"
CHANNELS = (RETRY, RAIL, CONTACT)

#: Day-of-month salary dates, weighted toward the 1st and 7th.
PAYDAY_DOM_WEIGHTS: dict[int, float] = {
    1: 0.40, 2: 0.08, 5: 0.10, 7: 0.15, 10: 0.07, 15: 0.05, 25: 0.06, 28: 0.09,
}


@dataclass(frozen=True, slots=True)
class WorldConstants:
    p_in_retry: float
    p_in_rail: float
    p_in_contact: float
    p_out: float
    attrition_horizon_days: float
    attrition_lambda: float

    def p_in(self, channel: str) -> float:
        return {
            RETRY: self.p_in_retry,
            RAIL: self.p_in_rail,
            CONTACT: self.p_in_contact,
        }[channel]


@dataclass(frozen=True, slots=True)
class WorldModel:
    constants: WorldConstants
    reasons: dict[str, dict[str, Any]]
    content_sha256: str

    def spec_for(self, reason: str) -> dict[str, Any] | None:
        return self.reasons.get(reason)


@dataclass(frozen=True, slots=True)
class RecoveryTruth:
    """When each channel can recover this payment. Executor-only."""

    payment_id: str
    retry_opens_at: str | None
    retry_closes_at: str | None
    rail_opens_at: str | None
    rail_closes_at: str | None
    contact_opens_at: str | None
    contact_closes_at: str | None
    blocker: str

    def window(self, channel: str) -> tuple[datetime, datetime] | None:
        opens = getattr(self, f"{channel}_opens_at")
        closes = getattr(self, f"{channel}_closes_at")
        if opens is None or closes is None:
            return None
        return datetime.fromisoformat(opens), datetime.fromisoformat(closes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "RecoveryTruth":
        return RecoveryTruth(**d)

    def all_timestamps(self) -> list[str]:
        """Every truth timestamp, for the leak assertions to hunt for."""
        return [
            v
            for v in (
                self.retry_opens_at, self.retry_closes_at,
                self.rail_opens_at, self.rail_closes_at,
                self.contact_opens_at, self.contact_closes_at,
            )
            if v is not None
        ]


def load_world(path: Path | str = DEFAULT_WORLD_PATH) -> WorldModel:
    import hashlib

    path = Path(path)
    raw = path.read_bytes()
    doc = tomllib.loads(raw.decode("utf-8"))

    c = doc["constants"]
    for key in ("p_in_retry", "p_in_rail", "p_in_contact", "p_out"):
        if not 0.0 <= float(c[key]) <= 1.0:
            raise ValueError(f"{path}: constants.{key} = {c[key]} is outside [0, 1]")
    if float(c["attrition_lambda"]) < 0:
        raise ValueError(f"{path}: attrition_lambda must be >= 0")
    if float(c["attrition_horizon_days"]) <= 0:
        raise ValueError(f"{path}: attrition_horizon_days must be > 0")

    constants = WorldConstants(
        p_in_retry=float(c["p_in_retry"]),
        p_in_rail=float(c["p_in_rail"]),
        p_in_contact=float(c["p_in_contact"]),
        p_out=float(c["p_out"]),
        attrition_horizon_days=float(c["attrition_horizon_days"]),
        attrition_lambda=float(c["attrition_lambda"]),
    )

    reasons = doc["reason"]
    valid_kinds = {"immediate", "immediate_if_not_upi", "delay", "payday", "outage_end", "never"}
    for reason, spec in reasons.items():
        for channel in CHANNELS:
            kind = spec.get(channel)
            if kind not in valid_kinds:
                raise ValueError(f"{path}: [reason.{reason}].{channel} = {kind!r} is not a valid kind")
            if kind == "delay" and ("delay_min_hours" not in spec or "delay_max_hours" not in spec):
                raise ValueError(f"{path}: [reason.{reason}] uses 'delay' but omits delay_min/max_hours")

    return WorldModel(
        constants=constants,
        reasons=reasons,
        content_sha256=hashlib.sha256(raw).hexdigest(),
    )


def next_payday(after: datetime, payday_dom: int) -> datetime:
    """First occurrence of `payday_dom` strictly after `after`, at 10:00 local."""
    year, month = after.year, after.month
    for _ in range(3):
        # Clamp to the month's length so a 28th/30th payday lands in February.
        import calendar

        dom = min(payday_dom, calendar.monthrange(year, month)[1])
        candidate = after.replace(
            year=year, month=month, day=dom, hour=10, minute=0, second=0, microsecond=0
        )
        if candidate > after:
            return candidate
        month += 1
        if month > 12:
            month, year = 1, year + 1
    raise AssertionError("failed to find a payday within three months")


def build_truth(
    *,
    payment_id: str,
    reason: str,
    method: str,
    created_at: datetime,
    payday_dom: int,
    outage_end: datetime | None,
    outage_label: str | None,
    world: WorldModel,
    stream,
) -> RecoveryTruth:
    """Derive one payment's hidden windows from the failure's semantics."""
    spec = world.spec_for(reason)
    if spec is None:
        raise ValueError(f"config/world.toml has no [reason.{reason}] section")

    horizon = created_at + timedelta(days=world.constants.attrition_horizon_days)
    opens: dict[str, datetime | None] = {}
    blocker = "terminal"

    for channel in CHANNELS:
        kind = spec[channel]
        if kind == "never":
            opens[channel] = None
        elif kind == "immediate":
            opens[channel] = created_at
        elif kind == "immediate_if_not_upi":
            opens[channel] = None if method == "upi" else created_at
        elif kind == "delay":
            hours = stream.uniform(float(spec["delay_min_hours"]), float(spec["delay_max_hours"]))
            opens[channel] = created_at + timedelta(hours=hours)
            if channel == RETRY:
                blocker = f"delay:{hours:.2f}h"
        elif kind == "payday":
            opens[channel] = next_payday(created_at, payday_dom)
            if channel == RETRY:
                blocker = f"payday:{opens[channel].date().isoformat()}"
        elif kind == "outage_end":
            opens[channel] = outage_end
            if channel == RETRY:
                blocker = f"outage:{outage_label}" if outage_label else "outage"
        else:  # pragma: no cover -- load_world validates the enum
            raise AssertionError(kind)

    # A window that would open after the customer is gone never opens at all.
    fields: dict[str, str | None] = {}
    for channel in CHANNELS:
        o = opens[channel]
        if o is None or o >= horizon:
            fields[f"{channel}_opens_at"] = None
            fields[f"{channel}_closes_at"] = None
        else:
            fields[f"{channel}_opens_at"] = o.isoformat()
            fields[f"{channel}_closes_at"] = horizon.isoformat()

    if all(fields[f"{c}_opens_at"] is None for c in CHANNELS):
        blocker = "unrecoverable"

    return RecoveryTruth(payment_id=payment_id, blocker=blocker, **fields)  # type: ignore[arg-type]


def pick_payday_dom(customer_id: str, seed: int) -> int:
    """Stable per-customer payday. Same customer, same payday, every run."""
    s = rng.corpus_stream(seed, f"payday|{customer_id}")
    doms = list(PAYDAY_DOM_WEIGHTS)
    return s.choices(doms, weights=[PAYDAY_DOM_WEIGHTS[d] for d in doms], k=1)[0]


def write_ground_truth(truths: list[RecoveryTruth], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for t in truths:
            fh.write(json.dumps(t.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def load_ground_truth(path: Path | str) -> dict[str, RecoveryTruth]:
    """Executor-only. selfcheck L5 fails the build if anything else imports this."""
    with Path(path).open("r", encoding="utf-8") as fh:
        truths = [RecoveryTruth.from_dict(json.loads(line)) for line in fh if line.strip()]
    return {t.payment_id: t for t in truths}


def ground_truth_path_for(seed: int, n: int, directory: Path | str) -> Path:
    return Path(directory) / f"ground_truth_v2_seed{seed}_n{n}.jsonl"
