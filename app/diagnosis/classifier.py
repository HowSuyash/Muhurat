"""The Classifier interface and its rules-based implementation.

Interface design notes (this is the Day-2 swap point, so the constraints are
deliberate):

1. `classify` takes the whole `RazorpayError`, not a code string. The error
   object carries `description` prose, `source` and `step` -- exactly the
   signal an LLM implementation needs. Passing only the code would guarantee
   the interface change this Protocol exists to prevent.

2. Synchronous. Every LLM SDK ships a sync client, and the orchestrator calls
   classifiers through a thread pool, so a slow implementation gets concurrency
   without touching this signature.

3. No provider concepts leak in: no message lists, no model or temperature
   kwargs, no client objects, no API keys. An implementation's configuration is
   its own constructor's business.

4. `Diagnosis.rule_id` is optional. A rules classifier fills it; an LLM
   classifier leaves it None and puts its justification in `reason`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.diagnosis.rules import RuleTable, load_rule_table
from app.models import Diagnosis, RazorpayError


@runtime_checkable
class Classifier(Protocol):
    """Maps a payment failure to a diagnosis. Implementation-agnostic."""

    name: str
    version: str

    def classify(self, error: RazorpayError) -> Diagnosis:
        """Diagnose a single failure. Must never raise for unrecognised input --
        return a UNKNOWN/low-confidence Diagnosis instead, so the pipeline can
        record the gap rather than crash on it."""
        ...


class RulesClassifier:
    """Deterministic lookup against config/rules.toml.

    Confidence is not a probability estimate -- it is a coverage signal. A
    matched rule is 1.0 because the mapping is definitional; the fallback is
    0.0 because the table genuinely has nothing to say. That makes
    `confidence == 0.0` a precise measure of where a rules table runs out, which
    is the quantity a Day-2 LLM arm has to improve on.
    """

    name = "rules"

    def __init__(self, table: RuleTable | None = None) -> None:
        self._table = table if table is not None else load_rule_table()
        # Version is derived from the table's content hash, so an audit record
        # pins the exact rules that produced it. Editing rules.toml changes the
        # recorded classifier version automatically.
        self.version = f"1.0+{self._table.content_sha256[:12]}"

    @property
    def table(self) -> RuleTable:
        return self._table

    def classify(self, error: RazorpayError) -> Diagnosis:
        rule, matched = self._table.lookup(error.reason)
        if matched:
            reason_text = f"[{rule.rule_id}] reason={error.reason!r} -> {rule.failure_class}. {rule.rationale}"
        else:
            reason_text = (
                f"[FALLBACK] reason={error.reason!r} has no rule in the table "
                f"(source={error.source!r}, step={error.step!r}). {rule.rationale}"
            )

        return Diagnosis(
            failure_class=rule.failure_class,
            recommended_action=rule.recommended_action,
            retriable=rule.retriable,
            min_wait_hours=rule.min_wait_hours,
            confidence=1.0 if matched else 0.0,
            reason=reason_text,
            classifier_name=self.name,
            classifier_version=self.version,
            rule_id=rule.rule_id if matched else None,
        )
