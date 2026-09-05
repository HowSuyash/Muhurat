"""Generate the synthetic failed-payment corpus and its hidden ground truth.

Two files come out of this, deliberately separate:

    data/corpus/failed_payments_v2_seed42_n300.jsonl   agent-visible
    data/corpus/ground_truth_v2_seed42_n300.jsonl      executor-only

Nothing but app/execution/window.py may load the second one. scripts/selfcheck.py
fails the build if any other module imports it.

Usage:
    python -m scripts.generate_corpus --seed 42 --n 300
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter

from app.corpus.generator import (
    DEFAULT_CORPUS_DIR,
    corpus_path_for,
    generate_corpus,
    write_corpus,
)
from app.corpus.truth import ground_truth_path_for, write_ground_truth
from app.diagnosis.classifier import RulesClassifier


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a seeded corpus of failed payments.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    payments, truths = generate_corpus(n=args.n, seed=args.seed)
    path = write_corpus(payments, corpus_path_for(args.seed, args.n))
    truth_path = write_ground_truth(
        truths, ground_truth_path_for(args.seed, args.n, DEFAULT_CORPUS_DIR)
    )

    classifier = RulesClassifier()
    classes = Counter(str(classifier.classify(p.error).failure_class) for p in payments)
    total = sum(p.amount_paise for p in payments)

    print(f"corpus     {path}")
    print(f"           sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")
    print(f"truth      {truth_path}")
    print(f"           sha256 {hashlib.sha256(truth_path.read_bytes()).hexdigest()}")
    print(f"records    {len(payments)}")
    print(f"at risk    Rs {total / 100:,.2f}")
    print(f"customers  {len({p.customer_id for p in payments})}")
    print()
    print("by failure class (as diagnosed by the rules classifier):")
    for name, count in classes.most_common():
        print(f"  {name:<20} {count:>4}  ({count / len(payments):5.1%})")
    print()
    print("hidden recovery channels (ground truth -- no arm can read this):")
    for channel in ("retry", "rail", "contact"):
        open_n = sum(1 for t in truths if getattr(t, f"{channel}_opens_at") is not None)
        print(f"  {channel:<10} opens for {open_n:>3} / {len(truths)} payments")
    print()
    print("root blocker:")
    for name, count in Counter(t.blocker.split(":")[0] for t in truths).most_common():
        print(f"  {name:<14} {count:>4}")


if __name__ == "__main__":
    main()
