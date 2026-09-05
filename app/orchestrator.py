"""Runs one arm (classifier + policy + executor) over a corpus.

The orchestrator is the only place the three interfaces meet, and it knows
nothing concrete about any of them -- swapping in an LLM classifier or the real
Razorpay client on Day 2 changes the constructor call, not this file.

Concurrency note: classification runs through a thread pool because a Day-2 LLM
classifier is I/O bound and would otherwise take minutes over 300 records. The
Classifier Protocol stays synchronous; the concurrency lives here. Results are
reassembled in corpus order, and outcomes use per-payment RNG streams, so the
thread pool cannot affect any measured number.
"""

from __future__ import annotations

import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.audit import DEFAULT_RUNS_DIR, AuditLog
from app.diagnosis.classifier import Classifier
from app.execution.base import PaymentExecutor
from app.models import (
    CUSTOMER_CONTACT_ACTIONS,
    GATEWAY_ATTEMPT_ACTIONS,
    AttemptStatus,
    DecisionRecord,
    FailedPayment,
    RecoveryRequest,
)
from app.policy.base import RecoveryPolicy


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_arm(
    payments: list[FailedPayment],
    classifier: Classifier,
    policy: RecoveryPolicy,
    executor: PaymentExecutor,
    *,
    seed: int,
    corpus_path: Path | str,
    arm: str | None = None,
    run_id: str | None = None,
    runs_dir: Path | str | None = None,
    max_workers: int = 8,
    enable_sqlite: bool = True,
) -> tuple[str, list[DecisionRecord]]:
    """Execute one arm end to end. Returns (run_id, decision records)."""
    arm = arm or policy.name
    run_id = run_id or f"{arm}-seed{seed}-{uuid.uuid4().hex[:8]}"

    log = AuditLog(
        run_id,
        runs_dir=runs_dir if runs_dir is not None else DEFAULT_RUNS_DIR,
        enable_sqlite=enable_sqlite,
    )

    # Diagnose everything up front. Classification does not depend on outcomes,
    # so this is safe to parallelise and keeps the attempt loop purely sequential.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        diagnoses = list(pool.map(lambda p: classifier.classify(p.error), payments))

    records: list[DecisionRecord] = []

    for payment, diagnosis in zip(payments, diagnoses, strict=True):
        attempt_number = 1
        # Attempts accumulate in wall-clock time from the original failure. The
        # policy chooses each gap; the executor sees only where that lands.
        clock = datetime.fromisoformat(payment.created_at)
        while True:
            decision = policy.decide(payment, diagnosis, attempt_number)
            if decision is None:
                break

            action = decision.action
            clock = clock + timedelta(hours=decision.delay_hours)
            outcome = executor.execute(
                RecoveryRequest(
                    payment=payment,
                    action=action,
                    diagnosis=diagnosis,
                    attempt_number=attempt_number,
                    run_seed=seed,
                    scheduled_at=clock.isoformat(),
                    delay_hours=decision.delay_hours,
                )
            )

            record = DecisionRecord(
                decision_id=f"{run_id}:{payment.payment_id}:{attempt_number}",
                run_id=run_id,
                ts=_now(),
                arm=arm,
                payment_id=payment.payment_id,
                customer_id=payment.customer_id,
                amount_paise=payment.amount_paise,
                method=str(payment.method),
                error_code=payment.error.code,
                error_reason=payment.error.reason,
                gateway_message=payment.error.description,
                error_source=payment.error.source,
                error_step=payment.error.step,
                institution=payment.institution,
                prior_attempt_count=payment.prior_attempt_count,
                attempt_number=attempt_number,
                scheduled_at=clock.isoformat(),
                delay_hours=decision.delay_hours,
                classifier_name=diagnosis.classifier_name,
                classifier_version=diagnosis.classifier_version,
                rule_id=diagnosis.rule_id,
                confidence=diagnosis.confidence,
                failure_class=str(diagnosis.failure_class),
                reason_text=diagnosis.reason,
                policy_name=policy.name,
                action_chosen=str(action),
                recommended_action=str(diagnosis.recommended_action),
                is_customer_contact=int(action in CUSTOMER_CONTACT_ACTIONS),
                is_gateway_attempt=int(action in GATEWAY_ATTEMPT_ACTIONS),
                executor_name=outcome.executor_name,
                outcome=str(outcome.status),
                outcome_error_reason=outcome.error_reason,
                amount_recovered_paise=outcome.amount_recovered_paise,
                success_probability=outcome.success_probability,
                latency_ms=outcome.latency_ms,
                rng_stream_key=outcome.rng_stream_key,
            )
            log.write(record)
            records.append(record)

            if outcome.status is AttemptStatus.SUCCEEDED:
                break
            if outcome.status is AttemptStatus.SUPPRESSED:
                # A suppression is terminal for this payment: the policy has
                # decided no further automated attempt is worth making.
                break

            attempt_number += 1

    corpus_path = Path(corpus_path)
    log.close(
        run_meta={
            "run_id": run_id,
            "arm": arm,
            "started_at": _now(),
            "seed": seed,
            "corpus_path": str(corpus_path),
            "corpus_sha256": _sha256_file(corpus_path),
            "rules_sha256": getattr(getattr(classifier, "table", None), "content_sha256", ""),
            "outcome_model_sha256": getattr(getattr(executor, "model", None), "content_sha256", ""),
            "ground_truth_sha256": (
                _sha256_file(executor.ground_truth_path)
                if getattr(executor, "ground_truth_path", None) is not None
                else ""
            ),
            "classifier_name": classifier.name,
            "classifier_version": classifier.version,
            "policy_name": policy.name,
            "executor_name": executor.name,
            "max_attempts": policy.max_attempts,
            "n_payments": len(payments),
        }
    )
    return run_id, records
