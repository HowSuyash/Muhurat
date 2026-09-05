"""Loads config/rules.toml into an indexed, validated rule table.

This module contains no payment logic. It has no if/else on error codes -- the
mapping lives entirely in the TOML file, which is why the README table can be
generated from it (scripts/render_rules_table.py) and shown to a panel as data.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

from app.models import FailureClass, RecoveryAction

DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "config" / "rules.toml"


@dataclass(frozen=True, slots=True)
class Rule:
    rule_id: str
    reason: str
    razorpay_code: str
    failure_class: FailureClass
    retriable: bool
    recommended_action: RecoveryAction
    min_wait_hours: float
    rationale: str


@dataclass(frozen=True, slots=True)
class RuleTable:
    rules: tuple[Rule, ...]
    fallback: Rule
    source_path: Path
    content_sha256: str

    _by_reason: dict[str, Rule] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_reason", {r.reason: r for r in self.rules})

    def lookup(self, reason: str) -> tuple[Rule, bool]:
        """Return (rule, matched). On no match, returns the fallback with False."""
        rule = self._by_reason.get(reason)
        if rule is None:
            return self.fallback, False
        return rule, True

    @property
    def covered_reasons(self) -> frozenset[str]:
        return frozenset(self._by_reason)

    def __len__(self) -> int:
        return len(self.rules)


def _build_rule(raw: dict, *, rule_id: str, reason: str, razorpay_code: str) -> Rule:
    try:
        failure_class = FailureClass(raw["failure_class"])
    except ValueError as exc:
        raise ValueError(f"rule {rule_id!r}: unknown failure_class {raw['failure_class']!r}") from exc
    try:
        action = RecoveryAction(raw["recommended_action"])
    except ValueError as exc:
        raise ValueError(f"rule {rule_id!r}: unknown recommended_action {raw['recommended_action']!r}") from exc

    wait = float(raw["min_wait_hours"])
    if wait < 0:
        raise ValueError(f"rule {rule_id!r}: min_wait_hours must be >= 0, got {wait}")
    rationale = str(raw["rationale"]).strip()
    if not rationale:
        raise ValueError(f"rule {rule_id!r}: rationale is required (it is the audit trail)")

    return Rule(
        rule_id=rule_id,
        reason=reason,
        razorpay_code=razorpay_code,
        failure_class=failure_class,
        retriable=bool(raw["retriable"]),
        recommended_action=action,
        min_wait_hours=wait,
        rationale=rationale,
    )


def load_rule_table(path: Path | str = DEFAULT_RULES_PATH) -> RuleTable:
    """Parse and validate the rules table. Raises on any malformed row."""
    path = Path(path)
    raw_bytes = path.read_bytes()
    doc = tomllib.loads(raw_bytes.decode("utf-8"))

    rules: list[Rule] = []
    seen_ids: set[str] = set()
    seen_reasons: set[str] = set()

    for raw in doc.get("rule", []):
        rule_id = raw["rule_id"]
        reason = raw["reason"]
        if rule_id in seen_ids:
            raise ValueError(f"duplicate rule_id {rule_id!r} in {path}")
        if reason in seen_reasons:
            raise ValueError(f"duplicate reason {reason!r} in {path} -- ambiguous mapping")
        seen_ids.add(rule_id)
        seen_reasons.add(reason)
        rules.append(_build_rule(raw, rule_id=rule_id, reason=reason, razorpay_code=raw["razorpay_code"]))

    if not rules:
        raise ValueError(f"{path} contains no rules")

    fb = doc["fallback"]
    fallback = _build_rule(fb, rule_id="FALLBACK", reason="*", razorpay_code="*")

    return RuleTable(
        rules=tuple(rules),
        fallback=fallback,
        source_path=path,
        content_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
