# Muhurat

### Recovery is a timing problem

*Muhurat* (मुहूर्त) is the auspicious moment to act — the hour chosen because acting then
succeeds and acting otherwise does not. Indian markets still open a ceremonial Muhurat trading
session on Diwali for exactly that reason.

That is this project's whole argument. A failed payment does not need more retries; it needs the
retry placed at the moment its real blocker clears. **The question is never whether to retry. It
is when, and on which channel.**

**Razorpay AI Buildathon, Track 03 — AI Revenue Recovery.**

A benchmark for failed-payment recovery. It ingests failed payments, diagnoses each one,
chooses a bounded intervention, executes it against a simulated world, and measures the money
actually recovered — against controls strong enough that beating them means something.

Every error code is real. Every number below is produced by the code in this repo and
reproduces offline with no API spend.

---

## See it running

**Live dashboard**
https://muhurat-27h5utppz-suyash25.vercel.app/

Six pages: the problem, the hidden-window mechanic (with a slider you can drag), a working
**advisor**, a **replay** of fourteen days in twenty seconds, the seven policies, and the
integrity evidence.

Or run it yourself:

```powershell
uvicorn app.main:app        # → http://127.0.0.1:8000
```

### The advisor — the winning policy with a front door

Give it any of Razorpay's 110 published error codes and it returns a decision: what went wrong,
whether it can be recovered at all, **when** to retry, what your system should do, and what to
say to the customer.

| `insufficient_funds` | |
|---|---|
| Verdict | **Wait for the salary cycle** |
| Action | `RETRY_AT_PAYDAY` · retry in ~3 days |
| Contacts customer | no — silent recovery |
| Do not | retry within minutes; the balance will not have moved |

It reads **only what a policy sees during the benchmark** — the error object, the rules table,
and the LLM for unmapped codes. It never touches `world.toml` or the ground truth, even though
both sit in this repo. That restraint is what makes its advice testable: because it uses exactly
what `llm_recommended` uses, the measured **58.11%** is the number this tool actually earns.

`GET /api/advise?reason=insufficient_funds&prior_attempts=2` returns the same thing as JSON.

### The replay — watch fourteen days pass

`naive_retry_3x` spends everything in the first six minutes and then flatlines for a fortnight.
`max_wait_probe` waits until day 13.9 and arrives to almost nothing. Each of the 300 payments is
a lamp that ignites when it is recovered.

---

## Result

300 failed payments, **₹509,681 at risk**, seed 42.

| Arm | Recovered | Rate | Payments | Gateway attempts | Customers touched | ₹ per customer touched |
|---|---:|---:|---:|---:|---:|---:|
| **`llm_recommended`** | **₹296,199** | **58.11%** | 174/300 | 428 | 42 | ₹7,052 |
| `llm_payday` | ₹295,749 | 58.03% | 174/300 | 429 | 42 | ₹7,042 |
| `rules_recommended` | ₹285,928 | 56.10% | 165/300 | 546 | 17 | **₹16,819** |
| `payday_inference` | ₹285,478 | 56.01% | 165/300 | 547 | 17 | ₹16,793 |
| `backoff_skip` | ₹159,405 | 31.28% | 101/300 | 642 | 0 | — |
| `naive_retry_3x` | ₹62,106 | 12.19% | 39/300 | 878 | 0 | — |
| `max_wait_probe` | ₹52,732 | 10.35% | 34/300 | 300 | 0 | — |

Theoretical ceiling (a perfect oracle that knows every hidden window): **71.57%**.

**The LLM adds ₹10,271 — +2.01 percentage points over the strongest non-AI control.** That is
the honest headline, and it is smaller than it would have been if the control were weaker. Read
the [Controls](#the-controls-are-deliberately-strong) section for why that is the point.

**That advantage is not robust.** A ±40% sweep over every world constant finds it holds in
17 of 21 scenarios and *inverts* in four — see [Sensitivity](#sensitivity-does-the-ranking-survive-our-assumptions-being-wrong).

**It does not win outright.** `llm_recommended` recovers more on *fewer* gateway attempts
(428 vs 546), but it gets there partly by contacting 25 more customers — and on
recovered-per-customer-touched it is **less than half as efficient** as the rules arm
(₹7,052 vs ₹16,819). If customer goodwill is priced at all, the rules arm may be the better
system. That tradeoff is reported rather than buried.

---

## What this benchmark measures

Most "payment recovery" demos measure whether a retry succeeded. That is uninteresting, because
the interesting decision is not *whether* to retry but **when**, and on **which channel**.

Each failed payment here has up to three hidden recovery channels, each with a window:

| Channel | Meaning |
|---|---|
| `retry` | the same instrument on the same rail, later |
| `rail` | move the payment to a different rail (UPI) |
| `contact` | ask the customer to act |

A window opens when the real blocker clears. A bank outage window opens when the outage ends. An
`insufficient_funds` window opens on that customer's **payday**. An `otp_attempts_exceeded`
window opens when the issuer's cooldown lapses. `card_expired` has no retry window at all — it
will never clear, no matter how patient you are.

An arm succeeds when its attempt lands inside an open window. **Timing is the decision
variable**, which is exactly what a naive benchmark cannot see.

### Why the earlier design was thrown away

Day 1 used a config file mapping `(failure_class, action) -> P(success)`. That design has a fatal
flaw: retry *timing* cannot affect the outcome, so the one place intelligence lives is invisible
to the measurement. Worse, fixing it by adding timing coefficients would mean hand-writing the
reward function that makes your own agent win.

`config/outcome_model.toml` and `app/execution/simulated.py` are kept as a legacy executor
(`--executor legacy`) so the two can be compared, but every number above comes from the
window executor.

### The integrity claim

> **No number in `config/world.toml` is keyed by failure class.**

There are exactly six global constants:

```toml
p_in_retry             = 0.90   # blocker gone -> the payment clears
p_in_rail              = 0.65   # an alternate rail exists and is enabled
p_in_contact           = 0.35   # the customer actually responds
p_out                  = 0.02   # softens the knife edge
attrition_horizon_days = 14     # after this the customer is gone
attrition_lambda       = 0.10   # per-day decay: lateness is paid for
```

Every difference between failure classes comes from *when the window opens*, which is derived
from the failure's own semantics — not from a coefficient chosen to favour an arm. `selfcheck`
asserts the file contains no failure-class name.

---

## The controls are deliberately strong

Three times during development, apparent "AI headroom" turned out to be a defect in our own
control. Each was found by comparing against the oracle ceiling, and each was fixed *before*
measuring the LLM:

| Defect | Was worth | What it actually was |
|---|---:|---|
| rail/contact re-rolled per attempt | ₹47,838 | executor bug — three contacts scored 0.96 instead of 0.35 |
| `card_expired` → contact, not rail | ₹26,276 | wrong action in the rules table |
| unmapped codes → `ESCALATE` | ₹79,023 | strawman fallback that recovered nothing |

**₹153,137** that the LLM arm would otherwise have banked without inferring anything. The
fallback one mattered most: escalating every unmapped code made the entire 36% tail score 0.0%
and look like a gap only AI could close. Four fallbacks were measured; the strongest was kept:

```
ESCALATE_MANUAL_REVIEW   Rs 206,905      RETRY_AFTER_BACKOFF 6h   Rs 280,351
REQUEST_NEW_INSTRUMENT   Rs 282,679      SWITCH_RAIL_TO_UPI       Rs 285,928  <- chosen
```

The arms, weakest to strongest:

- **`naive_retry_3x`** — retry everything immediately, 3×. Reach: 6 minutes. The floor.
- **`backoff_skip`** — exponential backoff, skip non-retriable. Reach: 21 hours. What a
  competent engineer builds first. Cannot reach a payday by construction.
- **`rules_recommended`** — follows the hand-authored rules table including per-reason waits and
  customer contact. The real bar.
- **`llm_recommended`** — identical policy to `rules_recommended`; **only the classifier
  differs**, so the gap is attributable to the model and not to a policy change.
- **`max_wait_probe`** — the degeneracy probe. See below.

---

## The degeneracy probe

If windows only closed at a fixed horizon, one attempt just before it would land inside *every*
window that exists and beat every arm with zero inference — replacing "the config decides the
answer" with "the horizon decides the answer".

`max_wait_probe` makes a single attempt at t+13.9 days. It records **140 in-window attempts** —
more than `backoff_skip` or `naive_retry_3x` — and still finishes **last** at 10.35%, because
`attrition_lambda` means a 13.9-day-old recovery keeps only ~25% of its value.

It runs in every benchmark, and `selfcheck` fails the build if it ever wins. *"What if I just
always wait two weeks?"* is answered by a number, not an argument.

---

## Where the LLM earns its place

`config/rules.toml` covers 35 of Razorpay's **110 published error reasons**. A hand-written table
cannot cover the rest, and in this corpus the tail is:

- **108 payments (36%)**, **51 distinct codes**, **₹187,814 at risk**
- spread across **16 semantic families** and **behaviourally mixed** — `rolling_limit` (₹41,450)
  has a real retry window, `instrument_dead` (₹27,070) is rail-recoverable, `merchant_config` is
  genuinely dead

So neither "escalate everything unmapped" nor "retry everything unmapped" works. The bucket has
to actually be told apart, and that is a generalisation problem no lookup table solves.

**The LLM wraps the rules path, it does not replace it.** Mapped codes use the rules table
unchanged and cost nothing; only unmapped codes reach the model. Diagnosis provenance for the
run above:

```
rules hits 192  |  cached verdicts 108  |  low-confidence fallbacks 3  |  errors 0  |  cache misses 0
```

Design constraints, all enforced by tests:

- **Fixed model** `claude-opus-5`. `temperature` is **not** set — sampling parameters were
  removed on the Claude 4.6+ family and return a 400. Determinism comes from the cache, which is
  a stronger guarantee than temperature 0 ever was.
- **Every verdict cached** to `data/llm_cache/`, keyed by SHA-256 of the prompt, and committed.
  The benchmark re-runs **offline with zero API spend**.
- **Never crashes, never silently guesses.** Low confidence (< 0.55), unparseable output, or any
  API error falls back to the rules classifier and logs it.
- **The prompt is built only from the Razorpay error object** — `reason`, `code`, `description`,
  `source`, `step`. No amount, customer, timestamp, outcome, or hidden window can reach it.

---

## Leak assertions

The integrity claim *is* the submission: if a policy could read the hidden windows, every number
here would be meaningless. **71 checks pass**, including thirteen written to be adversarial.

The line being defended is precise. *Inferring* a window from semantics is the skill under test —
a burst of `bank_technical_error` on one bank **should** predict an outage end. *Reading* a
window boundary off a visible field is cheating. L7 therefore asserts non-reconstruction, not
statistical independence.

| # | Assertion |
|---|---|
| L1 | `FailedPayment` has no window-shaped field |
| L2 | no window-shaped key at any depth in the visible corpus |
| L3 | no visible value equals a payment's *delayed* window boundary |
| L4 | object-graph walk from the objects a policy receives reaches no truth timestamp |
| L5 | AST scan: only the executor and reporters may import ground truth |
| L6 | no `Diagnosis` field — including free text — leaks a boundary |
| L7 | no single visible categorical field determines `retry_opens_at` |
| L8 | adversarial `getattr` probe over ~40 names finds nothing |
| L9 | the LLM prompt leaks no window, outcome, amount, customer or timestamp |
| L10 | every whitelisted field actually reaches the prompt |
| L11 | the cache key depends only on the reason, never on the payment |
| L12 | all 51 cached verdicts record their provenance |
| L13 | every LLM-arm diagnosis is a rule hit or a cached verdict — never a guess |

The truth lives in a **separate file** (`data/corpus/ground_truth_v2_*.jsonl`) that no classifier
or policy may import — a structural guarantee, not a naming convention.

**Common random numbers:** outcomes draw from streams keyed by
`(run_seed, payment_id, attempt_number)`, so two arms attempting the same payment at the same
attempt see identical luck. The measured delta is decisions, not sampling noise.

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

uvicorn app.main:app          # dashboard at http://127.0.0.1:8000
```

If `Activate.ps1` is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`

### Dashboard

`uvicorn app.main:app` serves a results dashboard at `http://127.0.0.1:8000` — the arm comparison
against the oracle ceiling, a per-arm breakdown of *why* each arm missed (too early, too late, or
no window ever existed), the per-class table, and the integrity checks. It is a viewer over
`data/runs/summary.json`, not a second source of truth, so the page and the CLI cannot disagree.
`GET /api/summary` returns the same JSON.

**No API key is needed, by design.** The LLM cache is committed, so the whole benchmark —
including the LLM arm — runs offline, deterministically, at zero cost. A reviewer can reproduce
every number in this README without an Anthropic account.

If you *do* have a key and want to verify the cache against a live model:
`python -m scripts.build_llm_cache --api` (51 calls, ~$0.15–0.40, one time), then re-run the
benchmark. The entries are keyed by prompt hash, so the comparison is direct.

The measurement pipeline is **pure stdlib** — `requirements.txt` is needed only for the FastAPI
stub, the settings module, and live API regeneration.

---

## Where the error codes come from

Razorpay publishes the authoritative reason list as a spreadsheet linked from their error docs.
It was downloaded, parsed, and committed to
[`data/reference/razorpay_error_reasons.csv`](data/reference/razorpay_error_reasons.csv) —
**110 distinct official `reason` values** with Razorpay's own explanation text.

- Source: `razorpay.com/docs/build/browser/assets/images/payments_error_reasons.xlsx`
- Schema: [About Error Codes](https://razorpay.com/docs/api/errors/) ·
  [Payment Method Error Parameters](https://razorpay.com/docs/errors/payment-methods/)

Nothing is invented — including `psp_app_ not_available`, whose embedded space is a typo in
Razorpay's own file, preserved verbatim so every code traces back to the published list.

---

## Architecture

```
corpus (visible)                     ground truth (executor-only, separate file)
      │                                            │
      ▼                                            │
Classifier ──► Diagnosis ──► RecoveryPolicy ──► PolicyDecision(action, delay_hours)
 RulesClassifier                                   │
 LLMClassifier ──► cache ──► claude-opus-5         ▼
                                            PaymentExecutor  ◄─── windows
                                          RecoveryWindowExecutor
                                                   │
                                                   ▼
                                         AuditLog → JSONL + SQLite
                                                   │
                                                   ▼
                                              evaluate.py
```

Three interfaces, three swap points — `Classifier`, `RecoveryPolicy`, `PaymentExecutor`. The
executor **never reads the action label**, only `delay_hours`, so a policy cannot win by
*calling* something `RETRY_AT_PAYDAY`. It has to estimate when payday actually is.

The rules table is **data, not `if/else`** — there is no branch on an error code anywhere in the
codebase. `python -m scripts.render_rules_table --write` regenerates the table below from
`config/rules.toml`, so the docs cannot drift.

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

---

## Sensitivity: does the ranking survive our assumptions being wrong?

The obvious objection to this benchmark is that its six world constants are invented. That is
fair — `p_in_retry = 0.90` is an informed estimate, not a measurement. The submission does not
claim those numbers are right; it claims the **ranking** is a real finding. `scripts/sensitivity.py`
tests that claim directly by perturbing every constant by ±40%, one at a time and jointly, and
re-running all six arms.

```
python -m scripts.sensitivity --joint 8
```

**Result: one claim holds, one does not.**

| Claim | Holds in |
|---|---|
| `max_wait_probe` never wins — waiting is not a substitute for reasoning | **21 / 21** |
| `llm_recommended` beats `rules_recommended` | **17 / 21** |

The degeneracy guarantee is solid under every perturbation tested. The LLM's advantage is not.
It inverts in four scenarios:

| Scenario | LLM | Rules | Delta | Winner |
|---|---:|---:|---:|---|
| `p_in_retry` −40% | ₹242,335 | ₹262,640 | **−₹20,305** | `payday_inference` |
| joint #3 | ₹243,247 | ₹266,124 | **−₹22,877** | `payday_inference` |
| joint #2 | ₹312,138 | ₹312,854 | −₹716 | `payday_inference` |
| `p_in_contact` −40% | ₹279,038 | ₹279,273 | −₹235 | `rules_recommended` |

Two of those are effectively ties. The two that are not share a cause: **when the retry channel
is materially weaker than assumed, the ordering flips and the payday arm wins.** The LLM arm
routes more payments to rail and contact, which is the right call under the baseline constants
and the wrong one when retry is cheap relative to them.

The honest reading: **+2.01pp is a result conditional on the world model, not a robust finding.**
A panel is entitled to weight it accordingly, and this section exists so they can. What *is*
robust is the shape of the benchmark — the probe never wins, and the ordering of the four
non-LLM arms is stable throughout.

---

## A negative result worth keeping

`payday_inference` was built to close the largest remaining gap: `INSUFFICIENT_FUNDS` is
₹93,217 at risk and the rules arm recovers ~52% where the oracle reaches 82%. The arm uses a
public prior over Indian salary dates (1st 40%, 7th 15%, month-end 9%) and aims its retries at
the most likely paydays, scored by probability × attrition decay. No API, no hidden data.

**It lost.** ₹285,478 vs ₹285,928 — worse by ₹450.

The diagnostics explain why, and the explanation is the interesting part:

| | `rules_recommended` | `payday_inference` |
|---|---:|---:|
| attempts landing in-window | 314 | **335** |
| missed by being too early | 50 | **42** |
| missed by being too late | 12 | **0** |
| recovered | **₹285,928** | ₹285,478 |

It is **strictly more accurate and slightly less valuable**. It hits the window more often and
is never late, but it gets there by waiting — and `attrition_lambda` charges for waiting. The
rules arm's crude "72 hours, then back off" guess misses more often, but when it lands it lands
early and keeps far more of the value.

Two honest conclusions. First, **precision without speed does not pay** in recovery: being right
on day 12 is worth ~30% of being right on day 1. Second, the ₹28,530 that the ceiling analysis
attributed to `INSUFFICIENT_FUNDS` was mostly *not* reachable by better retry timing — the oracle
earns it by choosing the **contact** channel, which this arm gives up when it commits to
`RETRY_AT_PAYDAY`. That is a flaw in the arm's design, not in the inference.

The arm ships in the benchmark rather than being deleted, because a result that disconfirms the
hypothesis is still a result, and `attrition_lambda` doing visible work is evidence the world
model is not a rubber stamp.

---

## Honest limitations

1. **The executor is simulated. No real Razorpay API call is made.** `RazorpayExecutor` was
   explicitly out of scope. Every rupee here is modelled, not settled.

2. **The LLM cache was not produced by live API calls — this is the biggest caveat in the
   project.** The 51 verdicts were produced by `claude-opus-5` reasoning over Razorpay's
   published descriptions in an interactive session, not through this repo's API path. Every
   cache entry records that in its `_provenance` field.

   **The same session also authored `config/world.toml`.** So the classifier and the world model
   share an author, and these verdicts are *not* a blind test. The +2.01pp should be read as an
   upper bound on what a live model would deliver, not an estimate of it. `scripts/build_llm_cache.py --api`
   regenerates the cache from a clean context and is the only way to settle it; it was not run
   because no API credential was available.

   What this does **not** affect: the window executor, the leak assertions, the controls, the
   degeneracy probe, and the sensitivity sweep are all independent of the cache. The four non-LLM
   arms and their ordering stand on their own.

3. **The window semantics are informed estimates, not measured data.** `config/world.toml`
   assigns each of the 110 codes to a semantic family. The families carry the reasoning and are
   reviewable, but a payments engineer could reasonably disagree with individual rows.

4. **The headline LLM gain inverts under 4 of 21 perturbations** — see the sensitivity
   section. It is conditional on the world model being roughly right about the retry channel.

5. **`payday_inference` underperforms and is retained anyway** — see the section above. It is
   reported as a negative result, not quietly dropped.

6. **The LLM's +2.01pp is within the range that corpus choices could move.** It is one seed and
   one corpus. A sensitivity sweep over the world constants (±40%) was planned and **not built**.

7. **The LLM arm is less contact-efficient** than the rules arm (₹7,052 vs ₹16,819 per customer
   touched). It buys part of its gain by bothering more people.

8. **`n=300`, one seed.** Per-class figures on small buckets (`MANDATE_FAILURE` n=10) carry large
   variance and should not be read as precise.

9. **Per-class comparison between the two top arms is not meaningful**, because the LLM
   *reclassifies* payments — the class buckets themselves differ between arms. Only the totals
   compare cleanly.

---

## Repo map

| Path | What it is |
|---|---|
| `config/world.toml` | recovery semantics for all 110 codes + the six global constants |
| `config/rules.toml` | the hand-authored rules table (35 codes), data not code |
| `app/corpus/truth.py` | hidden windows — **executor-only**, importing it elsewhere fails L5 |
| `app/execution/window.py` | the window executor |
| `app/diagnosis/llm_classifier.py` | LLM-over-rules classifier, cache, fallback |
| `data/llm_cache/` | 51 committed verdicts, keyed by prompt hash |
| `scripts/selfcheck.py` | 71 assertions incl. L1–L13 |
| `scripts/leakcheck.py` | the adversarial leak assertions |
| `scripts/sensitivity.py` | ±40% sweep over the world constants |
| `app/advisor.py` | the advisor: decision, timing, merchant + customer guidance |
| `app/static/dashboard.html` | the six-page dashboard |
| `app/main.py` | FastAPI: dashboard, `/api/advise`, `/api/replay`, `/health` |
| `scripts/export_replay.py` | flattens a finished run into a 109 KB event stream |

---

## Not built

- `RazorpayExecutor` / real API calls — explicitly out of scope.
