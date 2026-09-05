"""Generate world.toml entries for the long tail of Razorpay error reasons.

The head of the distribution (35 reasons) is hand-authored in config/world.toml
with a per-reason note. The tail is too large to hand-write and too important to
omit: Razorpay publishes 110 distinct reasons and a real merchant sees all of
them, so a corpus using only 35 flatters any hand-written rules table.

Rather than invent 75 individual stories, each tail reason is assigned to a
SEMANTIC FAMILY whose recovery behaviour follows from what the failure actually
is. The family carries the reasoning; the assignment is the reviewable claim.

Run once, then append the output to config/world.toml:
    python -m scripts.build_world_tail >> config/world.toml
"""

from __future__ import annotations

import csv
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CSV = ROOT / "data" / "reference" / "razorpay_error_reasons.csv"
WORLD = ROOT / "config" / "world.toml"

# ---------------------------------------------------------------------------
# Families. Each says: can the same instrument on the same rail EVER clear, and
# if so when; can another rail; can asking the customer.
# ---------------------------------------------------------------------------
FAMILIES: dict[str, dict] = {
    "transient_infra": dict(
        retry="delay", delay=(0.008, 0.133), rail="immediate", contact="immediate",
        code="GATEWAY_ERROR", source="gateway", step="payment_authorization",
        note="Transient infrastructure fault. Resolves in seconds to minutes without anyone doing anything.",
    ),
    "institution_outage": dict(
        retry="outage_end", rail="immediate_if_not_upi", contact="never",
        code="GATEWAY_ERROR", source="issuer_bank", step="payment_authorization",
        note="The institution is down. Clears when the outage clears; the customer cannot help.",
    ),
    "funds_cycle": dict(
        retry="payday", rail="never", contact="immediate",
        code="BAD_REQUEST_ERROR", source="issuer_bank", step="payment_authorization",
        note="Money arrives on a salary or statement cycle. Timing is the entire intervention.",
    ),
    "temporary_hold": dict(
        retry="delay", delay=(18.0, 36.0), rail="never", contact="immediate",
        code="BAD_REQUEST_ERROR", source="issuer_bank", step="payment_authorization",
        note="Funds are held rather than absent. The hold releases on its own within a day or so.",
    ),
    "rolling_limit": dict(
        retry="delay", delay=(12.0, 30.0), rail="immediate", contact="immediate",
        code="BAD_REQUEST_ERROR", source="issuer_bank", step="payment_authorization",
        note="A per-day or per-window cap. Resets at the next boundary -- reliably recoverable, just not soon.",
    ),
    "auth_friction": dict(
        retry="never", rail="immediate", contact="immediate",
        code="BAD_REQUEST_ERROR", source="customer", step="payment_authentication",
        note="The customer could not get through authentication. A silent retry cannot authenticate for them.",
    ),
    "auth_lockout": dict(
        retry="delay", delay=(8.0, 24.0), rail="immediate", contact="never",
        code="BAD_REQUEST_ERROR", source="issuer_bank", step="payment_authentication",
        note="Too many attempts; the issuer has locked the instrument. Only the cooldown lapsing helps.",
    ),
    "instrument_dead": dict(
        retry="never", rail="immediate", contact="immediate",
        code="BAD_REQUEST_ERROR", source="customer", step="payment_initiation",
        note="This instrument will never work again. Another rail or another instrument might.",
    ),
    "terminal_decline": dict(
        retry="never", rail="immediate", contact="immediate",
        code="BAD_REQUEST_ERROR", source="issuer_bank", step="payment_authorization",
        note="An opaque refusal. It will refuse again; route around it instead of repeating the question.",
    ),
    "risk_block": dict(
        retry="never", rail="never", contact="never",
        code="BAD_REQUEST_ERROR", source="issuer_bank", step="payment_authorization",
        note="A genuine risk signal that follows the customer across rails. Nothing automated should recover it.",
    ),
    "vpa_bad": dict(
        retry="never", rail="never", contact="immediate",
        code="BAD_REQUEST_ERROR", source="customer", step="payment_initiation",
        note="The UPI handle itself is wrong or restricted. Only a different handle from the customer helps.",
    ),
    "vpa_transient": dict(
        retry="delay", delay=(0.17, 1.5), rail="never", contact="immediate",
        code="GATEWAY_ERROR", source="network", step="payment_initiation",
        note="The UPI network failed to resolve, which is infrastructure -- not a bad handle.",
    ),
    "mandate_terminal": dict(
        retry="never", rail="never", contact="immediate",
        code="BAD_REQUEST_ERROR", source="bank", step="payment_authorization",
        note="The mandate is dead. Re-registration is the only path, and that needs the customer.",
    ),
    "mandate_transient": dict(
        retry="delay", delay=(1.0, 6.0), rail="immediate", contact="immediate",
        code="GATEWAY_ERROR", source="gateway", step="payment_authorization",
        note="A timeout during mandate setup is technical, not a refusal. It can still clear.",
    ),
    "merchant_config": dict(
        retry="never", rail="never", contact="never",
        code="BAD_REQUEST_ERROR", source="business", step="payment_initiation",
        note="A configuration problem on OUR side. No recovery action reaches it -- the business must fix it.",
    ),
    "input_invalid": dict(
        retry="never", rail="never", contact="immediate",
        code="BAD_REQUEST_ERROR", source="business", step="payment_initiation",
        note="The request itself was malformed. Retrying the same request reproduces the same error.",
    ),
    "reconciliation_lag": dict(
        retry="delay", delay=(2.0, 12.0), rail="never", contact="never",
        code="GATEWAY_ERROR", source="issuer_bank", step="payment_authorization",
        note="Settlement is still in flight. Often resolves itself -- and looks terminal while it does not.",
    ),
    "already_settled": dict(
        retry="never", rail="never", contact="never",
        code="BAD_REQUEST_ERROR", source="internal", step="payment_initiation",
        note="There is nothing to recover: already paid, duplicate, or no such record.",
    ),
    "beneficiary_bad": dict(
        retry="never", rail="never", contact="immediate",
        code="BAD_REQUEST_ERROR", source="beneficiary_bank", step="payment_authorization",
        note="The destination account cannot receive. Needs corrected details from the customer.",
    ),
    "credit_facility": dict(
        retry="never", rail="immediate", contact="immediate",
        code="BAD_REQUEST_ERROR", source="issuer", step="payment_initiation",
        note="A lending facility is unavailable or unapproved. Another rail can still take the payment.",
    ),
    "customer_abandoned": dict(
        retry="never", rail="immediate", contact="immediate",
        code="BAD_REQUEST_ERROR", source="customer", step="payment_authentication",
        note="The customer walked away. Re-engaging is the only route, and a cheaper rail helps.",
    ),
}

# ---------------------------------------------------------------------------
# Reason -> family. This mapping is the reviewable claim; a panel can disagree
# with any single row without the mechanism changing.
# ---------------------------------------------------------------------------
ASSIGNMENT: dict[str, list[str]] = {
    "transient_infra": [
        "gateway_technical_error", "request_timed_out", "payment_timed_out",
        "invalid_response_from_gateway", "server_error", "payment_collect_request_expired",
        "capture_failed",
    ],
    "institution_outage": [
        "bank_not_available", "bank_technical_error", "issuer_technical_error",
        "bank_cutoff_in_progress", "psp_not_available", "upi_app_technical_error",
        "payment_declined_due_to_high_traffic", "psp_app_ not_available",
    ],
    "funds_cycle": ["insufficient_funds", "credit_limit_exceeded"],
    "temporary_hold": ["funds_blocked_by_mandate"],
    "rolling_limit": [
        "transaction_daily_count_exceeded", "transaction_daily_limit_exceeded",
        "transaction_limit_exceeded", "transaction_frequency_limit_exceeded",
        "mcc_amount_limit_exceeded", "refund_limit_crossed",
    ],
    "auth_friction": [
        "otp_expired", "incorrect_otp", "authentication_failed", "payment_session_expired",
        "incorrect_cvv", "incorrect_atm_pin", "incorrect_pin", "incorrect_card_details",
        "incorrect_cardholder_name",
    ],
    "auth_lockout": ["otp_attempts_exceeded", "pin_attempts_exceeded"],
    "instrument_dead": [
        "card_expired", "incorrect_card_expiry_date", "debit_instrument_blocked",
        "debit_instrument_inactive", "card_not_enrolled", "card_number_invalid",
        "card_type_invalid", "pin_not_set",
    ],
    "terminal_decline": [
        "card_declined", "payment_declined", "debit_declined",
        "authorisation_declined_by_psp", "international_transaction_not_allowed",
        "payment_failed",
    ],
    "risk_block": ["payment_risk_check_failed", "compliance_violation"],
    "vpa_bad": [
        "invalid_vpa", "psp_not_registered", "transaction_on_vpa_restricted",
        "collect_on_mcc_blocked", "psp_app_not_supported",
    ],
    "vpa_transient": ["vpa_resolution_failed"],
    "mandate_terminal": [
        "mandate_creation_failed", "mandate_creation_declined", "mandate_creation_expired",
    ],
    "mandate_transient": ["mandate_creation_timeout", "reqauth_mandate_not_acknowledged"],
    "merchant_config": [
        "merchant_not_activated", "live_mode_not_enabled", "payment_method_not_enabled",
        "recurring_payment_not_enabled", "upi_collect_not_enabled", "upi_intent_not_enabled",
        "bank_not_enabled", "card_network_not_enabled", "upi_autopay_not_supported_on_psp",
        "user_not_eligible", "user_not_registered_for_netbanking",
    ],
    "input_invalid": [
        "input_validation_failed", "invalid_amount", "invalid_currency", "invalid_email",
        "invalid_mobile_number", "mobile_number_invalid", "invalid_user_details",
        "invalid_request", "invalid_order_id", "invalid_device",
        "amount_less_than_minimum_amount", "payment_amount_tampered",
        "order_amount_mismatch", "order_payment_method_mismatch",
        "mismatch_in_transaction_details",
    ],
    "reconciliation_lag": [
        "deemed_transaction", "credit_failed", "duplicate_rrn_found", "payment_pending",
        "collect_request_pending", "payment_pending_approval", "verification_failed",
        "bank_account_validation_failed",
    ],
    "already_settled": [
        "order_already_paid", "duplicate_request", "duplicate_refund_id", "record_not_found",
    ],
    "beneficiary_bad": [
        "beneficiary_account_does_not_exist", "beneficiary_account_dormant",
        "bank_account_invalid",
    ],
    "credit_facility": [
        "credit_limit_expired", "credit_limit_inactive", "credit_limit_not_approved",
        "credit_not_permitted", "emi_greater_than_max_amount", "emi_plan_unavailable",
    ],
    "customer_abandoned": ["payment_cancelled"],
}


def all_reasons() -> list[str]:
    seen: list[str] = []
    with REFERENCE_CSV.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            r = (row.get("Reason") or "").strip()
            if r and r not in seen:
                seen.append(r)
    return seen


def family_of() -> dict[str, str]:
    out = {}
    for fam, reasons in ASSIGNMENT.items():
        for r in reasons:
            if r in out:
                raise ValueError(f"{r} assigned to both {out[r]} and {fam}")
            out[r] = fam
    return out


def main() -> None:
    reasons = all_reasons()
    fam = family_of()

    missing = [r for r in reasons if r not in fam]
    extra = [r for r in fam if r not in reasons]
    if missing or extra:
        raise SystemExit(f"assignment incomplete.\n  missing: {missing}\n  not in CSV: {extra}")

    existing = tomllib.loads(WORLD.read_text(encoding="utf-8")).get("reason", {})
    todo = [r for r in reasons if r not in existing]

    print()
    print("# " + "=" * 74)
    print("# LONG TAIL -- generated by scripts/build_world_tail.py")
    print("#")
    print("# Razorpay publishes 110 distinct reasons. The head of the distribution is")
    print("# hand-authored above with a per-reason note; these are the remainder,")
    print("# assigned to semantic families so the reasoning is reviewable in one place")
    print("# rather than restated 75 times. The family assignment is the claim -- a")
    print("# panel can disagree with any single row without the mechanism changing.")
    print("#")
    print("# A hand-written rules table cannot cover this many codes, which is exactly")
    print("# the argument for a classifier that generalises to codes it has not seen.")
    print("# " + "=" * 74)

    for r in todo:
        f = FAMILIES[fam[r]]
        print()
        # Razorpay's own spreadsheet contains this typo; kept verbatim so every
        # code in this project traces back to their published list.
        quoted = f'"{r}"' if " " not in r else f'"{r}"  # sic: space is in the published file'
        print(f"[reason.{quoted.rstrip()}]" if " " not in r else f'[reason."{r}"]  # sic: typo is in Razorpay\'s file')
        print(f'retry = "{f["retry"]}"')
        if f["retry"] == "delay":
            print(f"delay_min_hours = {f['delay'][0]}")
            print(f"delay_max_hours = {f['delay'][1]}")
        print(f'rail = "{f["rail"]}"')
        print(f'contact = "{f["contact"]}"')
        print(f'family = "{fam[r]}"')
        print(f'note = "{f["note"]}"')


if __name__ == "__main__":
    main()
