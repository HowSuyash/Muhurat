# AI Revenue Recovery — Razorpay AI Buildathon, Track 03

Ingest failed payments, diagnose the root cause, choose a bounded recovery intervention,
execute it, and report the money actually recovered against a baseline.

**Day 1 status:** end-to-end pipeline running on 300 synthetic failures built from real
Razorpay error codes. Two baseline arms measured. No LLM involved yet — that is Day 2, and
the interfaces are already shaped for it.

---

## Day 1 result

300 failed payments, ₹509,681.00 at risk, seed 42.

| Arm | Recovered | Rate (by value) | Payments | Gateway attempts | Customers touched |
|---|---:|---:|---:|---:|---:|
| `naive_retry_3x` — retry everything, immediately, 3× | ₹128,315.00 | 25.18% | 74/300 | 794 | 0 |
| `backoff_skip` — exponential backoff, skip non-retriable | **₹219,757.00** | **43.12%** | 128/300 | **525** | 0 |

`backoff_skip` recovers **₹91,442 more while making 269 fewer gateway attempts**. It gets that
purely from two facts in the diagnosis: *can this failure ever clear on the same instrument*,
and *is waiting worth anything*. No AI required.

That is the point of including it. Beating `naive_retry_3x` proves nothing; **`backoff_skip` is
the number Day 2 has to beat.**

Where the difference comes from:

| Failure class | naive rate | backoff_skip rate | What changed |
|---|---:|---:|---|
| `BANK_DOWNTIME` | 25.7% | **98.7%** | Stopped retrying into the same outage window |
| `GATEWAY_TIMEOUT` | 71.4% | **98.3%** | Genuinely transient; a second look is cheap and works |
| `INSUFFICIENT_FUNDS` | 7.3% | **24.0%** | A balance does not change in the seconds between retries |
| `EXPIRED_CARD` | 0.0% | 0.0% | naive burned **60 attempts for ₹0**; backoff spent **0** |

---

## Quickstart (Windows / PowerShell)

```powershell
cd C:\Users\Admin\Documents\Razorpay
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python -m scripts.generate_corpus --seed 42 --n 300
python -m scripts.run_baseline    --seed 42 --n 300
python -m scripts.evaluate
python -m scripts.selfcheck
```

If `Activate.ps1` is blocked:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`

**The measurement pipeline is pure stdlib.** `requirements.txt` is needed only for the FastAPI
stub and the settings module, so the numbers above reproduce on a bare Python 3.11+ install
with nothing installed at all.

Optional API stub: `uvicorn app.main:app --reload` → `GET /health`.

---

## Where the error codes come from

Every error code in this project is real. Razorpay publishes the authoritative reason
enumeration as a spreadsheet linked from their error docs; it was downloaded, parsed, and
committed to [data/reference/razorpay_error_reasons.csv](data/reference/razorpay_error_reasons.csv)
— **114 official `reason` values** with Razorpay's own explanation text.

- Source: `razorpay.com/docs/build/browser/assets/images/payments_error_reasons.xlsx`
- Schema: [About Error Codes](https://razorpay.com/docs/api/errors/) ·
  [Payment Method Error Parameters](https://razorpay.com/docs/errors/payment-methods/)

Each synthetic record carries the **full** Razorpay error object — `code`, `description`,
`reason`, `source`, `step` — not just a code string. The gateway messages in the corpus are
Razorpay's own wording, verbatim. That matters for Day 2: an LLM classifier needs the prose.

---

## Architecture

```
corpus (300 records)
      │
      ▼
Classifier ──────────► Diagnosis (failure class, action, retriable, confidence, reason)
  RulesClassifier          │
  [Day 2: LLM]             ▼
                     RecoveryPolicy ──────► RecoveryAction
                       NaiveRetryPolicy         │
                       BackoffSkipPolicy        ▼
                       [Day 2: diagnosis-driven]
                                          PaymentExecutor
                                            SimulatedExecutor
                                            [Day 2: RazorpayExecutor]
                                                 │
                                                 ▼
                                          AuditLog → JSONL + SQLite
                                                 │
                                                 ▼
                                            evaluate.py
```

Three interfaces, three swap points:

| Interface | File | Today | Day 2 |
|---|---|---|---|
| `Classifier` | [app/diagnosis/classifier.py](app/diagnosis/classifier.py) | `RulesClassifier` | LLM-backed |
| `RecoveryPolicy` | [app/policy/base.py](app/policy/base.py) | naive, backoff-skip | diagnosis-driven |
| `PaymentExecutor` | [app/execution/base.py](app/execution/base.py) | `SimulatedExecutor` | `RazorpayExecutor` (test mode) |

### Why `classify` takes the whole error object

```python
class Classifier(Protocol):
    name: str
    version: str
    def classify(self, error: RazorpayError) -> Diagnosis: ...
```

Passing only `(error_code, gateway_message)` would throw away `description`, `source` and
`step` — exactly the signal an LLM needs — and force an interface change on Day 2. The
Protocol is also **synchronous** (every LLM SDK ships a sync client; the orchestrator supplies
concurrency via a thread pool), carries **no provider concepts** (no message lists, no model or
temperature kwargs, no clients, no keys), and makes `rule_id` **optional** so a non-rules
implementation leaves it `None` and justifies itself in `reason`.

---

## The rules table

Data, not `if/else`. There is no branch on an error code anywhere in the codebase —
[app/diagnosis/rules.py](app/diagnosis/rules.py) only loads and indexes
[config/rules.toml](config/rules.toml). The table below is generated from that file by
`python -m scripts.render_rules_table --write`, so it cannot drift.

<!-- BEGIN GENERATED RULES TABLE -->
_35 rules, generated from `config/rules.toml` (sha256 `a74575c07bfd`). Do not edit by hand._

| Rule | Razorpay `reason` | `error.code` | Failure class | Retriable | Recommended action | Wait | Rationale |
|---|---|---|---|---|---|---|---|
| `BD-001` | `bank_not_available` | `GATEWAY_ERROR` | BANK_DOWNTIME | yes | `RETRY_AFTER_BACKOFF` | 2h | Issuer outage windows run minutes to hours. An immediate retry lands inside the same outage. |
| `BD-002` | `bank_technical_error` | `GATEWAY_ERROR` | BANK_DOWNTIME | yes | `RETRY_AFTER_BACKOFF` | 2h | Core Banking System fault while processing. Transient, but not second-scale transient. |
| `BD-003` | `issuer_technical_error` | `GATEWAY_ERROR` | BANK_DOWNTIME | yes | `RETRY_AFTER_BACKOFF` | 2h | Technical fault at the issuer. Recovers on the issuer timeline, not ours. |
| `BD-004` | `bank_cutoff_in_progress` | `GATEWAY_ERROR` | BANK_DOWNTIME | yes | `RETRY_AFTER_BACKOFF` | 3h | Scheduled CBS cutoff. The most predictable failure in the whole table: it always ends. |
| `BD-005` | `psp_not_available` | `GATEWAY_ERROR` | BANK_DOWNTIME | yes | `RETRY_AFTER_BACKOFF` | 1h | PSP-side downtime. UPI infrastructure recovers faster than card rails, so a shorter wait. |
| `BD-006` | `upi_app_technical_error` | `GATEWAY_ERROR` | BANK_DOWNTIME | yes | `RETRY_AFTER_BACKOFF` | 1h | Fault in the customer own PSP app. Clears without any action from us. |
| `BD-007` | `payment_declined_due_to_high_traffic` | `GATEWAY_ERROR` | BANK_DOWNTIME | yes | `RETRY_AFTER_BACKOFF` | 1h | Bank saturated by a TPS surge. Retrying immediately adds load to the thing that is already failing. |
| `EC-001` | `card_expired` | `BAD_REQUEST_ERROR` | EXPIRED_CARD | **no** | `SWITCH_RAIL_TO_UPI` | — | Card is dead, but the customer is not. UPI works right now and asks less of them than re-entering a card. |
| `EC-002` | `incorrect_card_expiry_date` | `BAD_REQUEST_ERROR` | EXPIRED_CARD | **no** | `SWITCH_RAIL_TO_UPI` | — | Stored expiry is wrong. Rather than ask them to fix it, route around the card entirely. |
| `GT-001` | `gateway_technical_error` | `GATEWAY_ERROR` | GATEWAY_TIMEOUT | yes | `RETRY_NOW` | — | Transient infrastructure fault. This is the one class where retrying immediately is genuinely correct. |
| `GT-002` | `request_timed_out` | `GATEWAY_ERROR` | GATEWAY_TIMEOUT | yes | `RETRY_NOW` | — | No terminal state was ever reached, so nothing was actually declined. |
| `GT-003` | `payment_timed_out` | `GATEWAY_ERROR` | GATEWAY_TIMEOUT | yes | `RETRY_NOW` | — | Gateway returned no response. Retry is the only way to learn the real outcome. |
| `GT-004` | `invalid_response_from_gateway` | `GATEWAY_ERROR` | GATEWAY_TIMEOUT | yes | `RETRY_NOW` | — | Malformed response, not a decline. The customer instrument was never the problem. |
| `GT-005` | `server_error` | `SERVER_ERROR` | GATEWAY_TIMEOUT | yes | `RETRY_NOW` | — | Fault on the Razorpay side. Nothing about the payment itself is wrong. |
| `GT-006` | `payment_collect_request_expired` | `BAD_REQUEST_ERROR` | GATEWAY_TIMEOUT | yes | `RETRY_NOW` | — | The 5-10 minute UPI collect window lapsed. Re-sending the request costs nothing and often works. |
| `IF-001` | `insufficient_funds` | `BAD_REQUEST_ERROR` | INSUFFICIENT_FUNDS | yes | `RETRY_AT_PAYDAY` | 72h | An account balance changes on the salary cycle, not on a retry timer. Timing is the entire intervention. |
| `IF-002` | `credit_limit_exceeded` | `BAD_REQUEST_ERROR` | INSUFFICIENT_FUNDS | yes | `RETRY_AT_PAYDAY` | 72h | Cardless-EMI credit limit resets on the statement cycle. Same timing logic as a bank balance. |
| `IF-003` | `funds_blocked_by_mandate` | `BAD_REQUEST_ERROR` | INSUFFICIENT_FUNDS | yes | `RETRY_AFTER_BACKOFF` | 24h | A one-time-mandate hold is temporary. The funds reappear when the hold releases. |
| `MF-001` | `mandate_creation_failed` | `BAD_REQUEST_ERROR` | MANDATE_FAILURE | **no** | `REQUEST_NEW_MANDATE` | — | Terminal state per the Razorpay docs. A new mandate must be created; the old one is dead. |
| `MF-002` | `mandate_creation_declined` | `BAD_REQUEST_ERROR` | MANDATE_FAILURE | **no** | `REQUEST_NEW_MANDATE` | — | Declined by an entity in the chain. Terminal; re-registration is the only path. |
| `MF-003` | `mandate_creation_expired` | `BAD_REQUEST_ERROR` | MANDATE_FAILURE | **no** | `REQUEST_NEW_MANDATE` | — | The registration window closed. Nothing to retry against. |
| `MF-004` | `mandate_creation_timeout` | `GATEWAY_ERROR` | MANDATE_FAILURE | yes | `RETRY_AFTER_BACKOFF` | 4h | A timeout is a technical failure, not a refusal. Unlike MF-001 to MF-003 this one can still succeed. |
| `MF-005` | `reqauth_mandate_not_acknowledged` | `GATEWAY_ERROR` | MANDATE_FAILURE | yes | `RETRY_AFTER_BACKOFF` | 4h | The PSP never answered the authorisation request, which means PSP downtime rather than customer refusal. |
| `OT-001` | `otp_expired` | `BAD_REQUEST_ERROR` | OTP_TIMEOUT | yes | `SWITCH_RAIL_TO_UPI` | — | A fresh OTP needs the customer either way -- and UPI PIN entry completes far more often than card 3DS. |
| `OT-002` | `incorrect_otp` | `BAD_REQUEST_ERROR` | OTP_TIMEOUT | yes | `SWITCH_RAIL_TO_UPI` | — | They mistyped once and will again. Move them to the rail with the better completion rate. |
| `OT-003` | `authentication_failed` | `BAD_REQUEST_ERROR` | OTP_TIMEOUT | yes | `SWITCH_RAIL_TO_UPI` | — | 3DS abandonment is a friction problem. The fix is less friction, not another 3DS prompt. |
| `OT-004` | `otp_attempts_exceeded` | `BAD_REQUEST_ERROR` | OTP_TIMEOUT | yes | `RETRY_AFTER_BACKOFF` | 12h | Issuer has temporarily blocked the card. Contacting the customer now cannot help; the block must expire first. |
| `OT-005` | `payment_session_expired` | `BAD_REQUEST_ERROR` | OTP_TIMEOUT | yes | `SWITCH_RAIL_TO_UPI` | — | Session died, instrument is fine. Re-open on UPI rather than rebuild the card flow. |
| `RD-001` | `payment_risk_check_failed` | `BAD_REQUEST_ERROR` | RISK_DECLINE | **no** | `ESCALATE_MANUAL_REVIEW` | — | Retrying a risk decline raises the risk score further. The naive arm actively makes this one worse. |
| `RD-002` | `card_declined` | `BAD_REQUEST_ERROR` | RISK_DECLINE | yes | `SWITCH_RAIL_TO_UPI` | 6h | The issuer will not say why. Since we cannot diagnose it, change the rail instead of repeating the question. |
| `RD-003` | `payment_declined` | `BAD_REQUEST_ERROR` | RISK_DECLINE | yes | `SWITCH_RAIL_TO_UPI` | 6h | Opaque issuer or gateway decline. Same reasoning as card_declined: route around the silence. |
| `RD-004` | `international_transaction_not_allowed` | `BAD_REQUEST_ERROR` | RISK_DECLINE | **no** | `SWITCH_RAIL_TO_UPI` | — | This card can never clear this transaction. A domestic rail can. Retrying the card is a guaranteed loss. |
| `VP-001` | `invalid_vpa` | `BAD_REQUEST_ERROR` | INVALID_VPA | **no** | `REQUEST_NEW_INSTRUMENT` | — | The VPA is wrong or unregistered. It will be equally wrong on attempt three. Contrast VP-003. |
| `VP-002` | `psp_not_registered` | `BAD_REQUEST_ERROR` | INVALID_VPA | **no** | `REQUEST_NEW_INSTRUMENT` | — | No PSP registered on the customer device. Needs a different handle, not another attempt. |
| `VP-003` | `vpa_resolution_failed` | `GATEWAY_ERROR` | INVALID_VPA | yes | `RETRY_AFTER_BACKOFF` | 1h | NPCI resolution service failed -- an infrastructure fault, NOT a bad VPA. Looks like VP-001, opposite correct action. |
| _fallback_ | _(no match)_ | — | UNKNOWN | yes | `SWITCH_RAIL_TO_UPI` | — | No rule matched. When the cause is unknown, changing the rail is the best generic move: it sidesteps whatever the instrument-specific problem was without asking the customer for anything. |
<!-- END GENERATED RULES TABLE -->

### The pair worth pointing at

`invalid_vpa` (**never** retriable — the VPA is wrong) and `vpa_resolution_failed`
(**retriable** — NPCI's resolution service failed) are near-identical strings with opposite
correct actions. `naive_retry_3x` treats them identically and spends 62 attempts across the
class; `backoff_skip` spends 14 and recovers more (₹10,998 vs ₹9,679).

### `UNKNOWN` is deliberate

~4% of the corpus uses real Razorpay reasons intentionally left **out** of the rules table
(`deemed_transaction`, `mismatch_in_transaction_details`, `collect_on_mcc_blocked`, …). The
rules classifier bottoms out at `UNKNOWN` with `confidence = 0.0`. That is a measurable
coverage gap, and closing it is the argument for a Day-2 LLM classifier existing at all.

---

## The outcome model

Outcomes are probabilistic and grounded in the diagnosed failure class — an
`insufficient_funds` retry does not succeed at the same rate as a `gateway_timeout` retry.
Every probability lives in [config/outcome_model.toml](config/outcome_model.toml) **with a
comment explaining the reasoning**, so the assumptions are inspectable and arguable.

Immediate-retry success by class:

| Failure class | `RETRY_NOW` | Reasoning |
|---|---:|---|
| `GATEWAY_TIMEOUT` | 0.62 | Transient infra; the retry lands on a healthy node |
| `BANK_DOWNTIME` | 0.17 | The immediate retry hits the *same* outage window |
| `OTP_TIMEOUT` | 0.11 | A silent retry cannot produce an OTP |
| `MANDATE_FAILURE` | 0.09 | Only the timeout subset is transient |
| `INSUFFICIENT_FUNDS` | 0.06 | A balance does not change between retries |
| `RISK_DECLINE` | 0.03 | Repeat attempts *raise* the risk score |
| `INVALID_VPA` | 0.02 | A bad VPA is equally bad on attempt three |
| `EXPIRED_CARD` | 0.01 | Deterministic decline; near-total waste |

Two modifiers, also in config: per-class `attempt_decay`, and a `prior_attempt_penalty` of 0.90
per failure already on the record. The loader rejects any value outside `[0, 1]` and requires a
probability for **every** (class, action) pair, so a typo fails at startup rather than quietly
skewing the result.

**Honest limitation.** These probabilities are informed estimates, not measured data. They are
also keyed by *class*, which blends heterogeneous reasons: `INVALID_VPA`'s backoff rate of 0.21
is a blend across the whole class, but `backoff_skip` only ever retries the `vpa_resolution_failed`
subset, whose true rate is higher. The blend therefore **understates** any arm that correctly
selects the retriable subset — the reported `backoff_skip` figure is a conservative floor.

> **Day 3 (planned, not built):** sensitivity sweep at ±40% on every probability in
> `outcome_model.toml`, to show the *ranking of arms* is robust to these assumptions. That is
> the answer to "your numbers are invented" — the absolute rupees are a model, the ordering
> should not be.

---

## Why the two arms are comparable

Outcomes are drawn from streams keyed by `(run_seed, payment_id, attempt_number)`, not one
global RNG — see [app/rng.py](app/rng.py). Both arms attempting the same payment at the same
attempt number see the **identical** random draw, so the measured delta is attributable to the
decisions and not to sampling luck. This is common random numbers, and `selfcheck.py` asserts
it holds across every shared cell.

Consequences: the comparison is stable under reordering, parallelism, and adding or removing
arms. Actions that never touch the rails (`SUPPRESS_DO_NOT_RETRY`, `ESCALATE_MANUAL_REVIEW`)
consume no draw, so suppressing a payment in one arm cannot perturb another arm's luck.

Verified: two independent full runs produce byte-identical evaluation output.

---

## Customer-contact cost

Recovering more money by contacting every customer is not a better system. The eval counts
actions that consume a customer's attention (`PROMPT_CUSTOMER_OTP`, `REQUEST_NEW_INSTRUMENT`,
`REQUEST_NEW_MANDATE`) and reports **recovered per customer touched** alongside the headline.

Both Day-1 arms touch **zero** customers, by construction — neither has any basis for choosing
who to contact. That sets the denominator at zero and means every contact Day 2's arm makes has
to pay for itself.

---

## Audit trail

Every decision is one structured record, written to **both** `data/runs/<run_id>.jsonl`
(diffable, flushed per record so an interrupted run still leaves a trail) and SQLite at
`data/recovery.db` (queryable). Schema: [app/models.py](app/models.py) → `DecisionRecord`;
the SQLite DDL is generated from the dataclass fields so the two cannot drift.

`input → rule fired → reason → action chosen → outcome`, with the RNG stream key on every row
so any single outcome can be reproduced in isolation. A real record:

```json
{
  "error_reason": "incorrect_card_expiry_date",
  "gateway_message": "The customer has entered an incorrect expiry date of the card.",
  "error_source": "customer", "error_step": "payment_initiation",
  "rule_id": "EC-002", "confidence": 1.0,
  "failure_class": "EXPIRED_CARD",
  "reason_text": "[EC-002] reason='incorrect_card_expiry_date' -> EXPIRED_CARD. Wrong data on file. No amount of retrying corrects a stored value; only the customer can.",
  "recommended_action": "REQUEST_NEW_INSTRUMENT",
  "action_chosen": "RETRY_NOW",
  "policy_name": "naive_retry_3x",
  "outcome": "failed", "amount_recovered_paise": 0,
  "success_probability": 0.01,
  "rng_stream_key": "42|pay_0042000116|1"
}
```

`recommended_action: REQUEST_NEW_INSTRUMENT` next to `action_chosen: RETRY_NOW` is the naive
arm's failure made legible: the diagnosis was right and the policy ignored it.

The DB accumulates across runs — filter on `run_id` (or `arm`) when querying.

---

## Trusting the numbers

`python -m scripts.selfcheck` — **22 assertions, all passing.** Determinism for a fixed seed,
a different seed producing a different corpus, every probability in `[0, 1]`, complete
(class, action) coverage, every corpus reason either mapped or a known intended gap, the
`UNKNOWN` gap present at 3–10%, every record carrying a real Razorpay message, every payment
having an audit record, the 3-attempt cap respected, no attempt after a success, no payment
recovered twice, per-class sums reconciling to the total, recovered ≤ at risk, every audit row
fully populated, and common random numbers matching across arms.

---

## Corpus

300 records, seed 42, `sha256 52fc0c38…`. Method mix is UPI-dominant and failure classes are
conditioned on method, so no impossible rows exist (`invalid_vpa` only on UPI, `card_expired`
only on card, `mandate_creation_*` only on emandate). Amounts are lognormal ₹100–₹50,000 stored
in **paise**. Timestamps span 14 days on a diurnal curve; bank-downtime failures are clustered
into three synthetic outage windows rather than sprinkled uniformly, because that is what
downtime looks like and it is what makes retry *timing* matter. 151 distinct customers, so
repeat offenders exist. 0–2 prior attempts per record, skewed high for `insufficient_funds`.

---

## Secrets

`.env` holds `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` and is **git-ignored**. No Day-1 code
path reads either value — the simulated executor has no use for them. They are declared in
[app/settings.py](app/settings.py) as `SecretStr` so printing, logging, or dumping a `Settings`
object yields `**********`, ready for Day 2's real client. `/health` reports
`razorpay_credentials_configured: true|false` — presence only, never values.
[.env.example](.env.example) carries both keys, empty.

---

## Scope

**Day 1 (done):** skeleton · 300-record corpus from real codes · rules-based diagnosis ·
`Classifier` interface · two baseline arms · simulated outcomes · audit trail · eval harness ·
selfcheck.

**Explicitly not in Day 1:** no LLM calls, no AI SDK in `requirements.txt`, no API key read
anywhere, no dashboard, no real Razorpay calls, no auth, no Docker, no test suite beyond
`selfcheck.py`.

**Day 2:** `LLMClassifier` behind the existing Protocol · diagnosis-driven policy using the
full action vocabulary · `RazorpayExecutor` against test-mode APIs · three-way comparison.

**Day 3:** sensitivity sweep over `outcome_model.toml` · demo.
