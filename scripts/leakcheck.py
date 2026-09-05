"""L1-L8: prove no arm can read the hidden recovery windows.

The integrity claim IS the submission. If a policy could see when a window
opens, every result in this project would be meaningless -- so these are written
to be adversarial rather than reassuring.

The line being defended is subtle and worth stating precisely:

    INFERRING a window from semantics is the skill under test. A burst of
    `bank_technical_error` on one institution SHOULD predict an outage end --
    that is the whole point of the exercise.

    READING a window boundary off a visible field is cheating.

So L7 deliberately does NOT assert statistical independence. It asserts
non-reconstruction: no single visible field determines the answer.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import fields as dataclass_fields
from datetime import datetime
from pathlib import Path

from app.corpus.truth import RecoveryTruth
from app.models import FailedPayment

ROOT = Path(__file__).resolve().parents[1]

#: Anything named like a window is forbidden on the visible side.
FORBIDDEN_NAME = re.compile(
    r"window|truth|payday|outage|opens_at|closes_at|blocker|cooldown",
    re.IGNORECASE,
)

#: Importing these names means touching the ground-truth DATA. Path helpers
#: (`ground_truth_path_for`) and the public world constants (`load_world`) are
#: deliberately not on this list: knowing where the file lives, or what the
#: global constants are, reveals nothing about any individual payment.
FORBIDDEN_IMPORTS = {"load_ground_truth", "RecoveryTruth", "build_truth", "write_ground_truth"}

#: Only these modules may touch the ground truth. The executor needs it; the
#: generator writes it; the reporters read it after every arm has finished.
TRUTH_IMPORT_WHITELIST = {
    "app/execution/window.py",
    "app/corpus/generator.py",
    "app/corpus/truth.py",
    "scripts/generate_corpus.py",
    "scripts/evaluate.py",
    "scripts/selfcheck.py",
    "scripts/leakcheck.py",
    # Harness: regenerates a corpus + truth to sweep attrition_horizon_days,
    # which is baked into the windows at generation time. It runs arms but is
    # never itself a classifier or a policy.
    "scripts/sensitivity.py",
}

#: Directories that may NEVER be whitelisted, whatever the list above says.
#: A policy or classifier reading ground truth is the one failure this whole
#: project exists to rule out, so it is enforced structurally rather than by
#: remembering to keep the whitelist honest.
NEVER_WHITELISTABLE = ("app/policy/", "app/diagnosis/")

#: Attribute names an adversarial policy would try. L8 asserts none resolve.
PROBE_NAMES = [
    "truth", "_truth", "recovery_truth", "ground_truth", "_ground_truth",
    "window", "windows", "_window", "recovery_window", "retry_window",
    "retry_opens_at", "retry_closes_at", "rail_opens_at", "rail_closes_at",
    "contact_opens_at", "contact_closes_at", "opens_at", "closes_at",
    "blocker", "payday", "payday_dom", "outage", "outage_end", "outage_label",
    "cooldown", "cooldown_until", "recoverable_at", "world", "_world",
    "p_in", "p_out", "success_probability", "hidden", "_hidden", "secret",
    "_secret", "oracle", "_oracle", "answer", "_answer", "solution", "cheat",
]


def _walk_json(node, path="$"):
    """Yield (path, key, value) for every node in a decoded JSON document."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield path, k, v
            yield from _walk_json(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk_json(v, f"{path}[{i}]")


def _scalars(node):
    if isinstance(node, dict):
        for v in node.values():
            yield from _scalars(v)
    elif isinstance(node, list):
        for v in node:
            yield from _scalars(v)
    else:
        yield node


def _parse_ts(value):
    if not isinstance(value, str) or len(value) < 19:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _reachable_values(root, max_depth=6):
    """Every value reachable from an object graph, following real references."""
    seen_ids, out, frontier = set(), [], [(root, 0)]
    while frontier:
        obj, depth = frontier.pop()
        if depth > max_depth or id(obj) in seen_ids:
            continue
        seen_ids.add(id(obj))
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            out.append(obj)
            continue
        if isinstance(obj, dict):
            for k, v in obj.items():
                frontier.append((k, depth + 1))
                frontier.append((v, depth + 1))
            continue
        if isinstance(obj, (list, tuple, set, frozenset)):
            for v in obj:
                frontier.append((v, depth + 1))
            continue
        for attr in getattr(obj, "__slots__", ()) or ():
            if hasattr(obj, attr):
                frontier.append((getattr(obj, attr), depth + 1))
        for v in getattr(obj, "__dict__", {}).values():
            frontier.append((v, depth + 1))
    return out


def _secret_timestamps(truth, created_at: str, horizon_close: str | None) -> set[str]:
    """The boundaries that are actually SECRET.

    Not every boundary is. Two kinds are public by construction and matching
    them proves nothing:

      * a channel that opens "immediately" opens at `created_at`, which the
        agent can already see. That the contact channel opens at once is a
        semantic fact, not privileged information.
      * every window closes at created_at + attrition_horizon_days, and the
        horizon is a published constant in config/world.toml.

    The secret is WHEN A DELAYED WINDOW OPENS -- payday, outage end, cooldown.
    That is what these checks hunt for; anything looser would fail on facts the
    agent is entitled to know and would make the assertion meaningless.
    """
    secret = set()
    for channel in ("retry", "rail", "contact"):
        opens = getattr(truth, f"{channel}_opens_at")
        if opens is not None and opens != created_at:
            secret.add(opens)
    return secret


def run_leak_checks(check, corpus_path, truth_map, payments, diagnoses):
    """Run L1-L8. `check(name, condition, detail)` reports each result."""
    corpus_path = Path(corpus_path)
    created_by_pid = {p.payment_id: p.created_at for p in payments}
    truth_ts_by_pid = {
        pid: _secret_timestamps(t, created_by_pid.get(pid, ""), None)
        for pid, t in truth_map.items()
    }
    all_truth_ts = {ts for s in truth_ts_by_pid.values() for ts in s}

    # ---- L1: the visible dataclass has no window-shaped field -------------
    bad_fields = [f.name for f in dataclass_fields(FailedPayment) if FORBIDDEN_NAME.search(f.name)]
    check("L1 FailedPayment has no window-shaped field", not bad_fields, str(bad_fields))

    # ---- L2: no forbidden KEY at any depth in the visible corpus ----------
    raw = [
        json.loads(line)
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    bad_keys = sorted(
        {f"{p}.{k}" for rec in raw for p, k, _ in _walk_json(rec) if FORBIDDEN_NAME.search(k)}
    )
    check("L2 no window-shaped key at any depth in visible corpus", not bad_keys, str(bad_keys[:5]))

    # ---- L3: no VALUE anywhere equals a truth timestamp -------------------
    hits = []
    for rec in raw:
        pid = rec["payment_id"]
        mine = truth_ts_by_pid.get(pid, set())
        mine_dt = {d for d in (_parse_ts(t) for t in mine) if d is not None}
        for value in _scalars(rec):
            if isinstance(value, str) and value in mine:
                hits.append((pid, value))
                continue
            dt = _parse_ts(value)
            if dt is not None and any(abs((dt - m).total_seconds()) < 1.0 for m in mine_dt):
                hits.append((pid, value))
    check("L3 no visible value equals this payment's window boundary", not hits, str(hits[:3]))

    # ---- L4: nothing reachable from the objects a policy is handed --------
    leaked = []
    for payment, diagnosis in list(zip(payments, diagnoses))[:60]:
        mine = truth_ts_by_pid.get(payment.payment_id, set())
        if not mine:
            continue
        for value in _reachable_values(payment) + _reachable_values(diagnosis):
            if isinstance(value, str) and value in mine:
                leaked.append((payment.payment_id, value))
    check("L4 no truth timestamp reachable from FailedPayment/Diagnosis", not leaked, str(leaked[:3]))

    # ---- L5: only whitelisted modules import the truth loader -------------
    offenders = []
    # A whitelist entry under a forbidden directory is itself a bug: it would
    # mean someone granted a policy or classifier access to the answers.
    illegal_whitelist = [
        w for w in TRUTH_IMPORT_WHITELIST if w.startswith(NEVER_WHITELISTABLE)
    ]
    check(
        "L5a no policy or classifier is whitelisted for ground truth",
        not illegal_whitelist,
        str(illegal_whitelist),
    )

    for py in sorted([*(ROOT / "app").rglob("*.py"), *(ROOT / "scripts").rglob("*.py")]):
        rel = py.relative_to(ROOT).as_posix()
        if rel in TRUTH_IMPORT_WHITELIST and not rel.startswith(NEVER_WHITELISTABLE):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("corpus.truth"):
                hit = {a.name for a in node.names} & FORBIDDEN_IMPORTS
                if hit:
                    offenders.append(f"{rel}:{node.lineno} imports {sorted(hit)}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.endswith("corpus.truth"):
                        offenders.append(f"{rel}:{node.lineno} imports the module wholesale")
    check("L5 only the executor and reporters import ground truth", not offenders, str(offenders[:5]))

    # ---- L6: no classifier output carries a truth timestamp ---------------
    diag_leaks = []
    for payment, d in zip(payments, diagnoses):
        mine = truth_ts_by_pid.get(payment.payment_id, set())
        blob = " ".join(str(getattr(d, f.name)) for f in dataclass_fields(d))
        for ts in mine:
            if ts in blob:
                diag_leaks.append((payment.payment_id, ts))
    check("L6 no Diagnosis field leaks a window boundary", not diag_leaks, str(diag_leaks[:3]))

    # ---- L7: no single visible field DETERMINES the window ----------------
    # Correlation is expected and desired; exact reconstruction is not.
    by_pid = {r["payment_id"]: r for r in raw}
    determiners = []
    candidate_fields = ["institution", "created_at", "amount_paise", "customer_id", "method", "currency"]
    for field_name in candidate_fields:
        groups: dict[str, set] = {}
        n_contrib = 0
        for pid, rec in by_pid.items():
            opens = truth_map[pid].retry_opens_at
            if opens is None:
                continue
            n_contrib += 1
            groups.setdefault(str(rec.get(field_name)), set()).add(opens)
        if len(groups) <= 1:
            continue
        # Near-unique fields are identifiers, not features. `created_at` is
        # distinct per payment, so it trivially "determines" everything the way
        # a primary key does -- but only if you already hold the answer table.
        # Judge genuine categorical features only. The cardinality test is
        # against the payments that actually CONTRIBUTED a window, not the whole
        # corpus: when only half the corpus has a retry window, comparing
        # against the corpus size lets a per-record identifier slip through.
        if len(groups) > 0.5 * max(1, n_contrib):
            continue
        # A field determines the answer if EVERY one of its values maps to
        # exactly one opening time -- i.e. reading the field gives the answer.
        if all(len(v) == 1 for v in groups.values()):
            determiners.append(field_name)
    check(
        "L7 no single visible field determines retry_opens_at",
        not determiners,
        f"determined by {determiners}",
    )

    # ---- L8: adversarial getattr probe ------------------------------------
    found = []
    for payment, diagnosis in list(zip(payments, diagnoses))[:40]:
        mine = truth_ts_by_pid.get(payment.payment_id, set())
        for obj in (payment, diagnosis, payment.error):
            for name in PROBE_NAMES:
                got = getattr(obj, name, None)
                if got is None:
                    continue
                if isinstance(got, str) and (got in mine or got in all_truth_ts):
                    found.append((payment.payment_id, name))
                elif isinstance(got, RecoveryTruth):
                    found.append((payment.payment_id, name))
    check("L8 adversarial getattr probe finds no truth", not found, str(found[:3]))
