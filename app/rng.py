"""Deterministic, independent RNG streams.

Why this exists
---------------
If every arm drew from one global random stream, the naive arm and the
backoff-skip arm would consume different numbers of draws and end up seeing
different "luck" on the same payments. Any measured difference would then be a
mix of better decisions and sampling noise, and a panel would be right to
discount it.

Instead every outcome is drawn from a stream keyed by
`(run_seed, payment_id, attempt_number)`. Two arms attempting the same payment
for the same time see the *identical* random draw, so the measured delta is
attributable to the decisions alone. This is the standard common-random-numbers
variance reduction, and it costs one small module.

A consequence worth stating: the streams are stable under reordering,
parallelism, and adding or removing arms. Nothing about the comparison depends
on execution order.
"""

from __future__ import annotations

import hashlib
import random


def stream_key(run_seed: int, payment_id: str, attempt_number: int) -> str:
    """The exact string hashed into a stream. Recorded in every audit row."""
    return f"{run_seed}|{payment_id}|{attempt_number}"


def _seed_from_key(key: str) -> int:
    # SHA-256 rather than hash(): Python's str hash is salted per process, which
    # would silently destroy reproducibility across runs.
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def stream(run_seed: int, payment_id: str, attempt_number: int) -> random.Random:
    """An independent Random for one (payment, attempt) cell."""
    return random.Random(_seed_from_key(stream_key(run_seed, payment_id, attempt_number)))


def fixed_trait_stream(run_seed: int, payment_id: str, trait: str) -> random.Random:
    """A stream for a property of the payment that does NOT re-roll per attempt.

    Whether a customer has a usable alternate rail, or will respond to being
    contacted, is a fixed fact about that customer -- not a coin flip repeated
    on every attempt. Drawing those per-attempt would mean three rail switches
    beat one at 0.96 vs 0.65, rewarding repetition for its own sake and
    inflating the achievable ceiling by ~10 percentage points.

    Keyed without the attempt number, so every attempt on that channel sees the
    same draw. Still keyed by run_seed and payment_id, so common random numbers
    across arms is preserved exactly.
    """
    return random.Random(_seed_from_key(f"trait|{run_seed}|{payment_id}|{trait}"))


def corpus_stream(seed: int, label: str) -> random.Random:
    """A named stream for corpus generation.

    Separate labels keep amounts, timestamps and error selection independent, so
    changing one dimension of the generator does not reshuffle the others.
    """
    return random.Random(_seed_from_key(f"corpus|{seed}|{label}"))
