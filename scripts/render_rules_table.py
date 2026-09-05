"""Render config/rules.toml as a markdown table for the README.

The rules table is data, so the README table is generated from it rather than
maintained by hand. Run this after editing rules.toml.

Usage:
    python -m scripts.render_rules_table            # print to stdout
    python -m scripts.render_rules_table --write    # splice into README.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.diagnosis.rules import load_rule_table

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
BEGIN = "<!-- BEGIN GENERATED RULES TABLE -->"
END = "<!-- END GENERATED RULES TABLE -->"


def render() -> str:
    table = load_rule_table()
    lines = [
        f"_{len(table)} rules, generated from `config/rules.toml` "
        f"(sha256 `{table.content_sha256[:12]}`). Do not edit by hand._",
        "",
        "| Rule | Razorpay `reason` | `error.code` | Failure class | Retriable | Recommended action | Wait | Rationale |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(table.rules, key=lambda x: x.rule_id):
        wait = "—" if r.min_wait_hours == 0 else f"{r.min_wait_hours:g}h"
        lines.append(
            f"| `{r.rule_id}` | `{r.reason}` | `{r.razorpay_code}` | {r.failure_class} "
            f"| {'yes' if r.retriable else '**no**'} | `{r.recommended_action}` | {wait} | {r.rationale} |"
        )
    fb = table.fallback
    lines.append(
        f"| _fallback_ | _(no match)_ | — | {fb.failure_class} "
        f"| {'yes' if fb.retriable else '**no**'} | `{fb.recommended_action}` | — | {fb.rationale} |"
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="splice into README.md between markers")
    args = ap.parse_args()

    body = render()
    if not args.write:
        print(body)
        return

    text = README.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(f"README.md is missing the {BEGIN} / {END} markers")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    README.write_text(f"{head}{BEGIN}\n{body}\n{END}{tail}", encoding="utf-8")
    print(f"updated {README}")


if __name__ == "__main__":
    main()
