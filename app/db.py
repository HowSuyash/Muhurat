"""SQLite schema for the audit trail.

The DDL is generated from `DecisionRecord`'s dataclass fields, so the table and
the record type cannot drift apart. Everything is TEXT/INTEGER/REAL -- no ORM,
no migrations, nothing to go wrong on Day 1.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from app.models import DECISION_FIELDS, DecisionRecord

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "recovery.db"

_SQLITE_TYPE = {
    "amount_paise": "INTEGER",
    "prior_attempt_count": "INTEGER",
    "attempt_number": "INTEGER",
    "amount_recovered_paise": "INTEGER",
    "latency_ms": "INTEGER",
    "confidence": "REAL",
    "delay_hours": "REAL",
    "success_probability": "REAL",
    "is_customer_contact": "INTEGER",
    "is_gateway_attempt": "INTEGER",
}

_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id             TEXT PRIMARY KEY,
    arm                TEXT NOT NULL,
    started_at         TEXT NOT NULL,
    seed               INTEGER NOT NULL,
    corpus_path        TEXT NOT NULL,
    corpus_sha256      TEXT NOT NULL,
    rules_sha256       TEXT NOT NULL,
    outcome_model_sha256 TEXT NOT NULL,
    ground_truth_sha256  TEXT NOT NULL DEFAULT '',
    classifier_name    TEXT NOT NULL,
    classifier_version TEXT NOT NULL,
    policy_name        TEXT NOT NULL,
    executor_name      TEXT NOT NULL,
    max_attempts       INTEGER NOT NULL,
    n_payments         INTEGER NOT NULL
)
"""


def _decisions_ddl() -> str:
    cols = ",\n    ".join(f"{name} {_SQLITE_TYPE.get(name, 'TEXT')}" for name in DECISION_FIELDS)
    return (
        f"CREATE TABLE IF NOT EXISTS decisions (\n    {cols},\n"
        "    PRIMARY KEY (decision_id)\n)"
    )


def connect(path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_RUNS_DDL)
    conn.execute(_decisions_ddl())
    conn.execute("CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions(run_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_decisions_payment ON decisions(run_id, payment_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_decisions_class ON decisions(run_id, failure_class)")
    conn.commit()


def insert_decisions(conn: sqlite3.Connection, records: list[DecisionRecord]) -> None:
    placeholders = ", ".join("?" for _ in DECISION_FIELDS)
    sql = f"INSERT OR REPLACE INTO decisions ({', '.join(DECISION_FIELDS)}) VALUES ({placeholders})"
    rows = [tuple(getattr(r, f) for f in DECISION_FIELDS) for r in records]
    conn.executemany(sql, rows)
    conn.commit()


def insert_run(conn: sqlite3.Connection, meta: dict) -> None:
    cols = list(meta)
    sql = (
        f"INSERT OR REPLACE INTO runs ({', '.join(cols)}) "
        f"VALUES ({', '.join('?' for _ in cols)})"
    )
    conn.execute(sql, tuple(meta[c] for c in cols))
    conn.commit()
