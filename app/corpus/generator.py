"""Synthetic failed-payment corpus, built from real Razorpay error reasons.

Every `reason` and every `description` string here is verbatim from
data/reference/razorpay_error_reasons.csv, which was parsed from Razorpay's own
published error-reason spreadsheet. Nothing is invented.

Realism choices that matter for the result:

* Method mix reflects an Indian PG (UPI-dominant), and failure classes are
  conditioned on method -- `invalid_vpa` only appears on UPI, `card_expired`
  only on card, `mandate_creation_*` only on emandate. No impossible rows.
* Bank-downtime failures are clustered into a small number of outage windows
  rather than sprinkled uniformly, because that is what downtime looks like and
  it is what makes retry *timing* matter.
* ~6% of records use real reasons deliberately left out of the rules table, so
  the rules classifier has a measurable coverage gap for Day 2 to close.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app import rng
from app.corpus.truth import (
    RecoveryTruth,
    WorldModel,
    build_truth,
    load_world,
    pick_payday_dom,
)
from app.models import FailedPayment, PaymentMethod, PriorAttempt, RazorpayError

REFERENCE_CSV = Path(__file__).resolve().parents[2] / "data" / "reference" / "razorpay_error_reasons.csv"
DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"

#: Failure-class mix across the corpus. Weighted toward the classes that
#: actually dominate Indian PG failure volume.
CLASS_MIX: dict[str, float] = {
    "INSUFFICIENT_FUNDS": 0.22,
    "GATEWAY_TIMEOUT": 0.18,
    "BANK_DOWNTIME": 0.16,
    "RISK_DECLINE": 0.12,
    "OTP_TIMEOUT": 0.11,
    "INVALID_VPA": 0.09,
    "EXPIRED_CARD": 0.07,
    "MANDATE_FAILURE": 0.05,
}

#: (reason, razorpay_code, source, step, allowed methods) for each class.
#: Reasons are the match keys in config/rules.toml.
CLASS_REASONS: dict[str, list[tuple[str, str, str, str, tuple[PaymentMethod, ...]]]] = {
    "INSUFFICIENT_FUNDS": [
        ("insufficient_funds", "BAD_REQUEST_ERROR", "issuer_bank", "payment_authorization",
         (PaymentMethod.UPI, PaymentMethod.CARD, PaymentMethod.NETBANKING, PaymentMethod.EMANDATE)),
        ("credit_limit_exceeded", "BAD_REQUEST_ERROR", "issuer_bank", "payment_authorization",
         (PaymentMethod.CARD,)),
        ("funds_blocked_by_mandate", "BAD_REQUEST_ERROR", "issuer_bank", "payment_authorization",
         (PaymentMethod.UPI, PaymentMethod.EMANDATE)),
    ],
    "BANK_DOWNTIME": [
        ("bank_not_available", "GATEWAY_ERROR", "issuer_bank", "payment_authorization",
         (PaymentMethod.NETBANKING, PaymentMethod.CARD, PaymentMethod.UPI)),
        ("bank_technical_error", "GATEWAY_ERROR", "issuer_bank", "payment_authorization",
         (PaymentMethod.UPI, PaymentMethod.CARD, PaymentMethod.NETBANKING)),
        ("issuer_technical_error", "GATEWAY_ERROR", "issuer_bank", "payment_authorization",
         (PaymentMethod.CARD, PaymentMethod.NETBANKING)),
        ("bank_cutoff_in_progress", "GATEWAY_ERROR", "issuer_bank", "payment_initiation",
         (PaymentMethod.NETBANKING, PaymentMethod.EMANDATE)),
        ("psp_not_available", "GATEWAY_ERROR", "customer_psp", "payment_initiation",
         (PaymentMethod.UPI,)),
        ("upi_app_technical_error", "GATEWAY_ERROR", "customer_psp", "payment_authentication",
         (PaymentMethod.UPI,)),
        ("payment_declined_due_to_high_traffic", "GATEWAY_ERROR", "issuer_bank", "payment_authorization",
         (PaymentMethod.UPI, PaymentMethod.NETBANKING)),
    ],
    "EXPIRED_CARD": [
        ("card_expired", "BAD_REQUEST_ERROR", "customer", "payment_initiation", (PaymentMethod.CARD,)),
        ("incorrect_card_expiry_date", "BAD_REQUEST_ERROR", "customer", "payment_initiation",
         (PaymentMethod.CARD,)),
    ],
    "OTP_TIMEOUT": [
        ("otp_expired", "BAD_REQUEST_ERROR", "customer", "payment_authentication",
         (PaymentMethod.CARD, PaymentMethod.NETBANKING)),
        ("incorrect_otp", "BAD_REQUEST_ERROR", "customer", "payment_authentication",
         (PaymentMethod.CARD, PaymentMethod.NETBANKING, PaymentMethod.EMANDATE)),
        ("authentication_failed", "BAD_REQUEST_ERROR", "customer", "payment_authentication",
         (PaymentMethod.CARD, PaymentMethod.NETBANKING)),
        ("otp_attempts_exceeded", "BAD_REQUEST_ERROR", "customer", "payment_authentication",
         (PaymentMethod.CARD,)),
        ("payment_session_expired", "BAD_REQUEST_ERROR", "customer", "payment_authentication",
         (PaymentMethod.CARD, PaymentMethod.NETBANKING, PaymentMethod.UPI)),
    ],
    "GATEWAY_TIMEOUT": [
        ("gateway_technical_error", "GATEWAY_ERROR", "gateway", "payment_authorization",
         (PaymentMethod.UPI, PaymentMethod.CARD, PaymentMethod.NETBANKING, PaymentMethod.EMANDATE)),
        ("request_timed_out", "GATEWAY_ERROR", "gateway", "payment_initiation",
         (PaymentMethod.UPI, PaymentMethod.CARD, PaymentMethod.NETBANKING)),
        ("payment_timed_out", "GATEWAY_ERROR", "gateway", "payment_authorization",
         (PaymentMethod.UPI, PaymentMethod.CARD, PaymentMethod.NETBANKING)),
        ("invalid_response_from_gateway", "GATEWAY_ERROR", "gateway", "payment_authorization",
         (PaymentMethod.CARD, PaymentMethod.NETBANKING)),
        ("server_error", "SERVER_ERROR", "internal", "payment_initiation",
         (PaymentMethod.UPI, PaymentMethod.CARD, PaymentMethod.NETBANKING, PaymentMethod.EMANDATE)),
        ("payment_collect_request_expired", "BAD_REQUEST_ERROR", "customer", "payment_authentication",
         (PaymentMethod.UPI,)),
    ],
    "RISK_DECLINE": [
        ("payment_risk_check_failed", "BAD_REQUEST_ERROR", "issuer_bank", "payment_authorization",
         (PaymentMethod.CARD, PaymentMethod.UPI)),
        ("card_declined", "BAD_REQUEST_ERROR", "issuer_bank", "payment_authorization",
         (PaymentMethod.CARD,)),
        ("payment_declined", "BAD_REQUEST_ERROR", "issuer_bank", "payment_authorization",
         (PaymentMethod.UPI, PaymentMethod.CARD, PaymentMethod.NETBANKING)),
        ("international_transaction_not_allowed", "BAD_REQUEST_ERROR", "issuer_bank", "payment_initiation",
         (PaymentMethod.CARD,)),
    ],
    "INVALID_VPA": [
        ("invalid_vpa", "BAD_REQUEST_ERROR", "customer", "payment_initiation", (PaymentMethod.UPI,)),
        ("psp_not_registered", "BAD_REQUEST_ERROR", "customer", "payment_initiation", (PaymentMethod.UPI,)),
        ("vpa_resolution_failed", "GATEWAY_ERROR", "network", "payment_initiation", (PaymentMethod.UPI,)),
    ],
    "MANDATE_FAILURE": [
        ("mandate_creation_failed", "BAD_REQUEST_ERROR", "bank", "payment_authorization",
         (PaymentMethod.EMANDATE, PaymentMethod.UPI)),
        ("mandate_creation_declined", "BAD_REQUEST_ERROR", "customer", "payment_authentication",
         (PaymentMethod.EMANDATE, PaymentMethod.UPI)),
        ("mandate_creation_expired", "BAD_REQUEST_ERROR", "customer", "payment_authentication",
         (PaymentMethod.EMANDATE,)),
        ("mandate_creation_timeout", "GATEWAY_ERROR", "gateway", "payment_authorization",
         (PaymentMethod.EMANDATE, PaymentMethod.UPI)),
        ("reqauth_mandate_not_acknowledged", "GATEWAY_ERROR", "customer_psp", "payment_authentication",
         (PaymentMethod.UPI,)),
    ],
}

#: Real Razorpay reasons deliberately absent from config/rules.toml. These
#: produce UNKNOWN diagnoses, which is the coverage gap a Day-2 LLM classifier
#: has to close. Kept at ~6% of the corpus.
UNCOVERED_REASONS: list[tuple[str, str, str, str, tuple[PaymentMethod, ...]]] = [
    ("mismatch_in_transaction_details", "BAD_REQUEST_ERROR", "gateway", "payment_authorization",
     (PaymentMethod.UPI, PaymentMethod.CARD)),
    ("deemed_transaction", "GATEWAY_ERROR", "issuer_bank", "payment_authorization", (PaymentMethod.UPI,)),
    ("collect_on_mcc_blocked", "BAD_REQUEST_ERROR", "network", "payment_initiation", (PaymentMethod.UPI,)),
    ("transaction_daily_limit_exceeded", "BAD_REQUEST_ERROR", "issuer_bank", "payment_authorization",
     (PaymentMethod.UPI, PaymentMethod.CARD)),
    ("debit_instrument_blocked", "BAD_REQUEST_ERROR", "issuer_bank", "payment_authorization",
     (PaymentMethod.CARD,)),
    ("user_not_eligible", "BAD_REQUEST_ERROR", "business", "payment_initiation", (PaymentMethod.EMANDATE,)),
]

UNCOVERED_SHARE = 0.06  # v2 only; v3 uses TAIL_SHARE below

#: Share of records drawn from the LONG TAIL -- real Razorpay reasons that no
#: rule in config/rules.toml covers. Razorpay publishes 110 distinct reasons and
#: a hand-written table realistically covers only the head, so a corpus built
#: from 35 codes systematically flatters the rules-based arms. At 0.35 the tail
#: is a third of volume, which is what makes generalising to unseen codes a
#: load-bearing capability rather than a rounding error.
TAIL_SHARE = 0.35

#: Error metadata by semantic family (families are declared in config/world.toml
#: via scripts/build_world_tail.py). Kept here, not in world.toml, because this
#: is corpus construction -- world.toml is only about recovery semantics.
FAMILY_META: dict[str, tuple[str, str, str]] = {
    "transient_infra": ("GATEWAY_ERROR", "gateway", "payment_authorization"),
    "institution_outage": ("GATEWAY_ERROR", "issuer_bank", "payment_authorization"),
    "funds_cycle": ("BAD_REQUEST_ERROR", "issuer_bank", "payment_authorization"),
    "temporary_hold": ("BAD_REQUEST_ERROR", "issuer_bank", "payment_authorization"),
    "rolling_limit": ("BAD_REQUEST_ERROR", "issuer_bank", "payment_authorization"),
    "auth_friction": ("BAD_REQUEST_ERROR", "customer", "payment_authentication"),
    "auth_lockout": ("BAD_REQUEST_ERROR", "issuer_bank", "payment_authentication"),
    "instrument_dead": ("BAD_REQUEST_ERROR", "customer", "payment_initiation"),
    "terminal_decline": ("BAD_REQUEST_ERROR", "issuer_bank", "payment_authorization"),
    "risk_block": ("BAD_REQUEST_ERROR", "issuer_bank", "payment_authorization"),
    "vpa_bad": ("BAD_REQUEST_ERROR", "customer", "payment_initiation"),
    "vpa_transient": ("GATEWAY_ERROR", "network", "payment_initiation"),
    "mandate_terminal": ("BAD_REQUEST_ERROR", "bank", "payment_authorization"),
    "mandate_transient": ("GATEWAY_ERROR", "gateway", "payment_authorization"),
    "merchant_config": ("BAD_REQUEST_ERROR", "business", "payment_initiation"),
    "input_invalid": ("BAD_REQUEST_ERROR", "business", "payment_initiation"),
    "reconciliation_lag": ("GATEWAY_ERROR", "issuer_bank", "payment_authorization"),
    "already_settled": ("BAD_REQUEST_ERROR", "internal", "payment_initiation"),
    "beneficiary_bad": ("BAD_REQUEST_ERROR", "beneficiary_bank", "payment_authorization"),
    "credit_facility": ("BAD_REQUEST_ERROR", "issuer", "payment_initiation"),
    "customer_abandoned": ("BAD_REQUEST_ERROR", "customer", "payment_authentication"),
}

#: Substring -> the only method a reason can occur on. Everything else is drawn
#: from the normal method mix. Keeps impossible rows out of the corpus.
METHOD_HINTS: tuple[tuple[tuple[str, ...], PaymentMethod], ...] = (
    (("vpa", "upi", "psp", "collect_on_mcc", "intent", "autopay"), PaymentMethod.UPI),
    (("mandate", "recurring", "reqauth"), PaymentMethod.EMANDATE),
    (("card", "cvv", "atm_pin", "emi", "cardholder"), PaymentMethod.CARD),
    (("netbanking",), PaymentMethod.NETBANKING),
)


def method_for_reason(reason: str, stream) -> PaymentMethod:
    """The method a tail reason can plausibly occur on."""
    for needles, method in METHOD_HINTS:
        if any(n in reason for n in needles):
            return method
    methods = list(METHOD_WEIGHTS)
    return _weighted(stream, methods, [METHOD_WEIGHTS[m] for m in methods])

#: Method mix, used only to pick which reason variant a record gets when the
#: chosen class supports several methods.
METHOD_WEIGHTS: dict[PaymentMethod, float] = {
    PaymentMethod.UPI: 0.62,
    PaymentMethod.CARD: 0.24,
    PaymentMethod.NETBANKING: 0.09,
    PaymentMethod.EMANDATE: 0.05,
}

#: Relative volume by hour of day (IST), peaking at lunch and late evening.
DIURNAL_WEIGHTS: tuple[float, ...] = (
    0.2, 0.1, 0.1, 0.1, 0.2, 0.4, 0.8, 1.2, 1.8, 2.4, 3.0, 3.6,
    4.2, 4.0, 3.2, 2.8, 2.6, 2.8, 3.4, 4.4, 5.0, 4.6, 2.6, 1.0,
)

CORPUS_START = datetime(2026, 8, 18, 0, 0, tzinfo=timezone.utc)
CORPUS_DAYS = 14

#: Institutions handling each payment. Visible on the record, because
#: correlated failures across one institution in a time neighbourhood are how
#: downtime is actually detected. Inferring an outage from these is the skill
#: under test; reading the outage end from the truth file is cheating.
BANKS: tuple[str, ...] = ("HDFC", "ICICI", "SBI", "AXIS", "KOTAK")
PSPS: tuple[str, ...] = ("@okhdfcbank", "@ybl", "@paytm", "@okaxis", "@apl")

#: How many synthetic outages the corpus contains. Each is scoped to one
#: institution, so "which bank is down" is answerable from visible data.
N_OUTAGES = 3


def load_reason_descriptions(path: Path | str = REFERENCE_CSV) -> dict[str, str]:
    """reason -> Razorpay's own explanation text, used as the gateway message."""
    out: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            reason = (row.get("Reason") or "").strip()
            if reason and reason not in out:
                out[reason] = (row.get("Explanation") or "").strip()
    return out


def _weighted(stream, options: list, weights: list[float]):
    return stream.choices(options, weights=weights, k=1)[0]


def _outage_windows(seed: int) -> list[tuple[str, datetime, datetime, str]]:
    """Synthetic outages, each scoped to one institution.

    Returns (institution, start, end, label). Downtime clusters in time AND in
    one institution -- that pairing is what makes it inferable from the visible
    corpus without ever reading a window boundary.
    """
    s = rng.corpus_stream(seed, "outages")
    windows = []
    for i in range(N_OUTAGES):
        start = CORPUS_START + timedelta(
            days=s.randrange(CORPUS_DAYS), hours=s.randrange(24), minutes=s.randrange(60)
        )
        end = start + timedelta(hours=s.uniform(1.5, 5.0))
        institution = BANKS[i % len(BANKS)]
        windows.append((institution, start, end, f"{institution}#{i}"))
    return sorted(windows, key=lambda w: w[1])


def _timestamp(stream, hour_stream) -> datetime:
    day = stream.randrange(CORPUS_DAYS)
    hour = _weighted(hour_stream, list(range(24)), list(DIURNAL_WEIGHTS))
    return CORPUS_START + timedelta(
        days=day, hours=hour, minutes=stream.randrange(60), seconds=stream.randrange(60)
    )


def _amount_paise(stream) -> int:
    """Lognormal, clamped to Rs.100 - Rs.50,000, rounded to the nearest rupee."""
    rupees = stream.lognormvariate(mu=6.75, sigma=1.15)
    rupees = min(50_000.0, max(100.0, rupees))
    return int(round(rupees)) * 100


def generate_corpus(
    n: int = 300, seed: int = 42, world: WorldModel | None = None
) -> tuple[list[FailedPayment], list[RecoveryTruth]]:
    """Build the corpus and its hidden ground truth together.

    Returns (agent-visible payments, executor-only truths). The two are written
    to SEPARATE files so no classifier or policy can reach the windows -- see
    app/corpus/truth.py and the L1-L8 assertions in scripts/selfcheck.py.
    """
    descriptions = load_reason_descriptions()
    world = world if world is not None else load_world()

    s_class = rng.corpus_stream(seed, "class")
    s_reason = rng.corpus_stream(seed, "reason")
    s_method = rng.corpus_stream(seed, "method")
    s_amount = rng.corpus_stream(seed, "amount")
    s_time = rng.corpus_stream(seed, "time")
    s_hour = rng.corpus_stream(seed, "hour")
    s_cust = rng.corpus_stream(seed, "customer")
    s_prior = rng.corpus_stream(seed, "prior")
    s_gap = rng.corpus_stream(seed, "gap")
    s_inst = rng.corpus_stream(seed, "institution")
    s_window = rng.corpus_stream(seed, "window")

    outages = _outage_windows(seed)
    classes = list(CLASS_MIX)
    class_weights = [CLASS_MIX[c] for c in classes]

    # The long tail: every published reason that no rule covers. Weighted with a
    # Zipf-like decay so a handful of tail codes are common and most are rare --
    # which is what a real tail looks like, and what makes memorising it futile.
    from app.diagnosis.rules import load_rule_table

    covered = load_rule_table().covered_reasons
    tail_reasons = [r for r in sorted(world.reasons) if r not in covered]
    tail_weights = [1.0 / (i + 1) ** 0.6 for i in range(len(tail_reasons))]
    s_tail = rng.corpus_stream(seed, "tail")

    n_customers = max(1, int(n * 0.6))  # ~180 customers over 300 records
    payments: list[FailedPayment] = []
    pending: list[tuple[str, str, str, datetime, str, datetime | None, str | None]] = []

    for i in range(n):
        if tail_reasons and s_class.random() < TAIL_SHARE:
            # A reason no rule covers. The rules classifier will bottom out at
            # its fallback here; a classifier that generalises need not.
            reason = _weighted(s_tail, tail_reasons, tail_weights)
            family = world.reasons[reason].get("family", "terminal_decline")
            code, source, step = FAMILY_META[family]
            method = method_for_reason(reason, s_method)
            failure_class = "TAIL"
        else:
            failure_class = _weighted(s_class, classes, class_weights)
            variants = CLASS_REASONS[failure_class]
            reason, code, source, step, methods = s_reason.choice(variants)
            # Pick a method the chosen reason can actually occur on, weighted by
            # the real method mix rather than uniformly.
            weights = [METHOD_WEIGHTS[m] for m in methods]
            method = _weighted(s_method, list(methods), weights)

        created = _timestamp(s_time, s_hour)
        outage_end: datetime | None = None
        outage_label: str | None = None

        if failure_class == "BANK_DOWNTIME":
            # Pull downtime failures into a real outage window, scoped to one
            # institution. The failure inherits that institution, so a burst of
            # failures on one bank in one time neighbourhood is visible signal.
            institution, start, end, label = outages[s_time.randrange(len(outages))]
            span = (end - start).total_seconds()
            created = start + timedelta(seconds=s_time.uniform(0, span))
            outage_end, outage_label = end, label
            if method is PaymentMethod.UPI:
                # UPI still names a PSP; keep the bank identity recognisable.
                institution = PSPS[BANKS.index(institution) % len(PSPS)]
        else:
            institution = (
                s_inst.choice(PSPS) if method is PaymentMethod.UPI else s_inst.choice(BANKS)
            )

        # Prior failures already on the record. Insufficient-funds customers
        # retry by hand more than anyone else, so they skew high.
        if failure_class == "INSUFFICIENT_FUNDS":
            n_prior = _weighted(s_prior, [0, 1, 2], [0.45, 0.35, 0.20])
        elif failure_class in ("EXPIRED_CARD", "INVALID_VPA"):
            n_prior = _weighted(s_prior, [0, 1, 2], [0.55, 0.30, 0.15])
        else:
            n_prior = _weighted(s_prior, [0, 1, 2], [0.72, 0.22, 0.06])

        priors: list[PriorAttempt] = []
        for k in range(n_prior):
            gap = timedelta(hours=s_gap.uniform(2, 96) * (n_prior - k))
            priors.append(
                PriorAttempt(
                    ts=(created - gap).isoformat(),
                    reason=reason,
                    status="failed",
                )
            )

        payment_id = f"pay_{seed:04d}{i:06d}"
        customer_id = f"cust_{s_cust.randrange(n_customers):05d}"
        payments.append(
            FailedPayment(
                payment_id=payment_id,
                customer_id=customer_id,
                amount_paise=_amount_paise(s_amount),
                currency="INR",
                method=method,
                created_at=created.isoformat(),
                error=RazorpayError(
                    code=code,
                    description=descriptions.get(reason, ""),
                    reason=reason,
                    source=source,
                    step=step,
                    metadata={"payment_id": payment_id},
                ),
                institution=institution,
                prior_attempts=tuple(priors),
            )
        )
        pending.append(
            (payment_id, reason, str(method), created, customer_id, outage_end, outage_label)
        )

    # Sort by failure time so the corpus reads like an event log.
    payments.sort(key=lambda p: (p.created_at, p.payment_id))

    # Build the hidden windows in the SAME sorted order, so the truth file lines
    # up with the corpus for anyone reading them side by side.
    order = {p.payment_id: idx for idx, p in enumerate(payments)}
    pending.sort(key=lambda t: order[t[0]])

    truths: list[RecoveryTruth] = []
    for payment_id, reason, method_str, created, customer_id, outage_end, outage_label in pending:
        truths.append(
            build_truth(
                payment_id=payment_id,
                reason=reason,
                method=method_str,
                created_at=created,
                payday_dom=pick_payday_dom(customer_id, seed),
                outage_end=outage_end,
                outage_label=outage_label,
                world=world,
                stream=s_window,
            )
        )

    return payments, truths


def write_corpus(payments: list[FailedPayment], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for p in payments:
            fh.write(json.dumps(p.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def read_corpus(path: Path | str) -> list[FailedPayment]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [FailedPayment.from_dict(json.loads(line)) for line in fh if line.strip()]


def corpus_path_for(
    seed: int, n: int, directory: Path | str = DEFAULT_CORPUS_DIR, version: str = "v2"
) -> Path:
    """v1 is Day 1's frozen corpus; v2 adds institutions and a truth file."""
    stem = "failed_payments" if version == "v1" else f"failed_payments_{version}"
    return Path(directory) / f"{stem}_seed{seed}_n{n}.jsonl"
