"""LLM classifier for the unmapped long tail.

WHAT THIS IS FOR
----------------
config/rules.toml covers 35 of Razorpay's 110 published error reasons. The other
75 are real codes a merchant genuinely sees, and 51 of them appear in the corpus
— 36% of records, Rs 187,814 at risk. A hand-written table cannot be extended to
cover them all, and the ones it misses are behaviourally mixed: some have a real
retry window, some are rail-recoverable, some are terminal. Guessing one
behaviour for the whole bucket is wrong in every direction.

That is the gap this closes, and it is the only place in the project where an
LLM earns its keep.

HOW IT COMPOSES
---------------
This does NOT replace RulesClassifier — it wraps it:

    rules match  -> use the rule, no API call, no cost
    rules miss   -> ask the model, cache the answer
    low conf     -> use the rules fallback instead
    API error    -> use the rules fallback, and log it

So every mapped code behaves exactly as before. Only the tail changes, which
makes the arm comparison a clean measurement of what the model adds.

DETERMINISM
-----------
Every answer is cached to disk keyed by a SHA-256 of the exact prompt, and the
cache is committed. The benchmark therefore re-runs offline with zero API spend
and identical results. Note that `temperature` is NOT set: it was removed from
the Claude 4.6+ model family and returns a 400 on claude-opus-5. Determinism
here comes from the cache, not from a sampling parameter — which is the stronger
guarantee, since temperature 0 was never a determinism promise either.

WHAT THE MODEL CAN SEE
----------------------
Only the fields of the Razorpay error object: reason, code, description, source,
step. That is a deliberately tighter surface than the pipeline allows — no
amount, no timestamp, no customer, no attempt history, and above all nothing
about the outcome or the hidden recovery window. `PROMPT_VISIBLE_FIELDS` below
is the whitelist, and scripts/selfcheck.py asserts the built prompt contains
nothing outside it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from app.diagnosis.classifier import RulesClassifier
from app.models import Diagnosis, FailureClass, RazorpayError, RecoveryAction

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "llm_cache"

#: Fixed model string. Never varied — a different model would invalidate every
#: cached answer and silently change the benchmark.
MODEL = "claude-opus-5"

#: Below this, the model's answer is discarded and the rules fallback is used.
#: An LLM that is unsure is worth less than a known-mediocre default.
MIN_CONFIDENCE = 0.55

#: The ONLY payment-derived fields that may reach the prompt. Everything else —
#: amount, timestamp, customer, prior attempts, and every outcome or window
#: field — is withheld. selfcheck asserts this.
PROMPT_VISIBLE_FIELDS: frozenset[str] = frozenset(
    {"reason", "code", "description", "source", "step"}
)

_ACTIONS = [a.value for a in RecoveryAction]
_CLASSES = [c.value for c in FailureClass]

SYSTEM_PROMPT = """You are a payments recovery analyst working with Razorpay error codes.

Given one Razorpay payment failure reason and its official description, decide how a \
merchant should try to recover that payment.

Think about the physical reality behind the error:
- Can the SAME instrument on the SAME rail ever succeed later? If so, what has to \
change first, and roughly how long does that take? A bank outage clears in hours. A \
salary credit takes days. An expired card never clears.
- Would moving the payment to a different rail (UPI) sidestep the problem?
- Would asking the customer to act (re-authenticate, supply a new instrument, \
re-register a mandate) fix it?
- Some failures are the merchant's own configuration problem, or are already settled, \
and no recovery action reaches them at all.

Be honest about uncertainty. If the description is ambiguous, give a lower confidence \
— a low-confidence answer is discarded in favour of a safe default, which is better \
than a confident wrong one."""


@dataclass(frozen=True, slots=True)
class LLMVerdict:
    failure_class: str
    recommended_action: str
    retriable: bool
    min_wait_hours: float
    confidence: float
    reasoning: str


def build_prompt(error: RazorpayError) -> str:
    """The exact user prompt. Also the cache key input.

    Only PROMPT_VISIBLE_FIELDS appear here. Kept as a standalone function so
    selfcheck can build a prompt and assert nothing else leaked into it.
    """
    return (
        f"Razorpay payment failure.\n\n"
        f"reason: {error.reason}\n"
        f"error code: {error.code}\n"
        f"description: {error.description}\n"
        f"source: {error.source}\n"
        f"step: {error.step}\n\n"
        f"Choose one failure_class from: {', '.join(_CLASSES)}\n"
        f"Choose one recommended_action from: {', '.join(_ACTIONS)}\n\n"
        f"Also give:\n"
        f"- retriable: true only if the same instrument on the same rail can eventually succeed\n"
        f"- min_wait_hours: how long to wait before the first attempt is worth making (0 if immediate)\n"
        f"- confidence: 0.0-1.0, how sure you are\n"
        f"- reasoning: one sentence a payments engineer would accept"
    )


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "failure_class": {"type": "string", "enum": _CLASSES},
        "recommended_action": {"type": "string", "enum": _ACTIONS},
        "retriable": {"type": "boolean"},
        "min_wait_hours": {"type": "number", "minimum": 0},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string"},
    },
    "required": [
        "failure_class", "recommended_action", "retriable",
        "min_wait_hours", "confidence", "reasoning",
    ],
    "additionalProperties": False,
}


def cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class LLMCache:
    """One JSON file per prompt hash. Committed to the repo."""

    def __init__(self, directory: Path | str = DEFAULT_CACHE_DIR) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> LLMVerdict | None:
        p = self.path_for(key)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        try:
            return LLMVerdict(
                failure_class=d["failure_class"],
                recommended_action=d["recommended_action"],
                retriable=bool(d["retriable"]),
                min_wait_hours=float(d["min_wait_hours"]),
                confidence=float(d["confidence"]),
                reasoning=d.get("reasoning", ""),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def put(self, key: str, verdict: LLMVerdict, *, reason: str, provenance: str) -> None:
        payload = {
            "reason": reason,
            "failure_class": verdict.failure_class,
            "recommended_action": verdict.recommended_action,
            "retriable": verdict.retriable,
            "min_wait_hours": verdict.min_wait_hours,
            "confidence": verdict.confidence,
            "reasoning": verdict.reasoning,
            "_model": MODEL,
            "_provenance": provenance,
        }
        self.path_for(key).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def __len__(self) -> int:
        return len(list(self.dir.glob("*.json")))

    def by_reason(self) -> dict[str, LLMVerdict]:
        """Index the cache by error reason instead of by prompt hash.

        The benchmark keys on a SHA-256 of the exact prompt, which is right for
        reproducibility: a changed prompt must miss. But an interactive caller
        types a reason without the corpus's exact description, source and step,
        so it would miss every time and silently fall back to rules.

        Since each cache entry records the reason it answers, indexing by that
        is the same data under a different key -- not a looser guarantee. Used
        only by app/advisor.py; the benchmark path is untouched.
        """
        out: dict[str, LLMVerdict] = {}
        for f in self.dir.glob("*.json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                r = d.get("reason")
                if not r:
                    continue
                out[r] = LLMVerdict(
                    failure_class=d["failure_class"],
                    recommended_action=d["recommended_action"],
                    retriable=bool(d["retriable"]),
                    min_wait_hours=float(d["min_wait_hours"]),
                    confidence=float(d["confidence"]),
                    reasoning=d.get("reasoning", ""),
                )
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return out


class LLMClassifier:
    """Rules first, model only for what the rules cannot place.

    Satisfies the same `Classifier` Protocol as RulesClassifier — the
    orchestrator, policies and audit schema are untouched.
    """

    name = "llm+rules"

    def __init__(
        self,
        rules: RulesClassifier | None = None,
        cache: LLMCache | None = None,
        *,
        allow_api: bool = False,
        min_confidence: float = MIN_CONFIDENCE,
        match_by_reason: bool = False,
    ) -> None:
        self._rules = rules if rules is not None else RulesClassifier()
        self._cache = cache if cache is not None else LLMCache()
        self._allow_api = allow_api
        self._min_confidence = min_confidence
        self.version = f"1.0+{MODEL}+{self._rules.version.split('+')[-1]}"
        # Counters so the run can report exactly what the model contributed.
        self.stats = {"rules_hit": 0, "cache_hit": 0, "api_call": 0,
                      "low_confidence": 0, "error_fallback": 0, "cache_miss": 0}
        # Off by default so the benchmark keeps its strict prompt-hash keying.
        self._by_reason = self._cache.by_reason() if match_by_reason else None

    @property
    def table(self):
        """Exposed so orchestrator run-metadata hashes the rules table as usual."""
        return self._rules.table

    def classify(self, error: RazorpayError) -> Diagnosis:
        base = self._rules.classify(error)

        # Mapped code: the rules table is authoritative and free. No API call.
        if base.rule_id is not None:
            self.stats["rules_hit"] += 1
            return base

        prompt = build_prompt(error)
        key = cache_key(prompt)
        verdict = self._cache.get(key)

        # Interactive callers (app/advisor.py) supply a reason without the
        # corpus's exact prose, so the prompt hash misses. Fall back to the
        # reason index, which is the same cached answer under another key.
        if verdict is None and self._by_reason is not None:
            verdict = self._by_reason.get(error.reason)
            if verdict is not None:
                self.stats["cache_hit"] += 1
                return self._from_verdict(verdict, error)

        if verdict is not None:
            self.stats["cache_hit"] += 1
        elif self._allow_api:
            verdict = self._call_api(prompt, key, error)
        else:
            self.stats["cache_miss"] += 1
            log.warning("llm: cache miss for reason=%r and API disabled; using rules fallback", error.reason)
            return self._fallback(base, "cache miss, API disabled")

        if verdict is None:
            return self._fallback(base, "API error or unparseable output")

        if verdict.confidence < self._min_confidence:
            self.stats["low_confidence"] += 1
            log.info("llm: confidence %.2f below %.2f for reason=%r; using rules fallback",
                     verdict.confidence, self._min_confidence, error.reason)
            return self._fallback(base, f"model confidence {verdict.confidence:.2f} too low")

        try:
            failure_class = FailureClass(verdict.failure_class)
            action = RecoveryAction(verdict.recommended_action)
        except ValueError:
            log.warning("llm: unknown enum in verdict for reason=%r; using rules fallback", error.reason)
            return self._fallback(base, "model returned an unknown class or action")

        return Diagnosis(
            failure_class=failure_class,
            recommended_action=action,
            retriable=verdict.retriable,
            min_wait_hours=max(0.0, verdict.min_wait_hours),
            confidence=verdict.confidence,
            reason=f"[LLM] reason={error.reason!r} unmapped by the rules table. {verdict.reasoning}",
            classifier_name=self.name,
            classifier_version=self.version,
            rule_id=None,  # no rule fired; the justification is in `reason`
        )

    def _from_verdict(self, verdict: LLMVerdict, error: RazorpayError) -> Diagnosis:
        """Build a Diagnosis from a cached verdict, applying the confidence floor."""
        if verdict.confidence < self._min_confidence:
            self.stats["low_confidence"] += 1
            return self._fallback(self._rules.classify(error),
                                  f"model confidence {verdict.confidence:.2f} too low")
        try:
            fc = FailureClass(verdict.failure_class)
            act = RecoveryAction(verdict.recommended_action)
        except ValueError:
            return self._fallback(self._rules.classify(error), "unknown class or action")
        return Diagnosis(
            failure_class=fc,
            recommended_action=act,
            retriable=verdict.retriable,
            min_wait_hours=max(0.0, verdict.min_wait_hours),
            confidence=verdict.confidence,
            reason=f"[LLM] reason={error.reason!r} unmapped by the rules table. {verdict.reasoning}",
            classifier_name=self.name,
            classifier_version=self.version,
            rule_id=None,
        )

    def _fallback(self, base: Diagnosis, why: str) -> Diagnosis:
        """Never crash, never silently guess — degrade to rules and say so."""
        return Diagnosis(
            failure_class=base.failure_class,
            recommended_action=base.recommended_action,
            retriable=base.retriable,
            min_wait_hours=base.min_wait_hours,
            confidence=base.confidence,
            reason=f"[LLM->rules fallback: {why}] {base.reason}",
            classifier_name=self.name,
            classifier_version=self.version,
            rule_id=None,
        )

    def _call_api(self, prompt: str, key: str, error: RazorpayError) -> LLMVerdict | None:
        """One live call. Only reached when allow_api=True and the cache misses."""
        try:
            import anthropic
        except ImportError:
            self.stats["error_fallback"] += 1
            log.error("llm: anthropic SDK not installed; using rules fallback")
            return None

        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=MODEL,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": RESPONSE_SCHEMA,
                    }
                },
            )
            if response.stop_reason == "refusal":
                self.stats["error_fallback"] += 1
                log.error("llm: model refused for reason=%r", error.reason)
                return None
            text = "".join(b.text for b in response.content if b.type == "text")
            data = json.loads(text)
            verdict = LLMVerdict(
                failure_class=data["failure_class"],
                recommended_action=data["recommended_action"],
                retriable=bool(data["retriable"]),
                min_wait_hours=float(data["min_wait_hours"]),
                confidence=float(data["confidence"]),
                reasoning=data.get("reasoning", ""),
            )
        except Exception as exc:  # noqa: BLE001 — a failed run must never crash the benchmark
            self.stats["error_fallback"] += 1
            log.error("llm: call failed for reason=%r (%s); using rules fallback", error.reason, exc)
            return None

        self.stats["api_call"] += 1
        self._cache.put(
            key, verdict, reason=error.reason,
            provenance=f"live {MODEL} API call via scripts/build_llm_cache.py",
        )
        return verdict
