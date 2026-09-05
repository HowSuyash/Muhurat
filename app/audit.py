"""Structured decision logging.

Every decision the pipeline makes is written to two sinks:

  * JSONL at data/runs/<run_id>.jsonl -- diffable, greppable, survives a
    schema change, and is what you hand a judge who asks "show me the trail".
  * SQLite -- queryable, so the eval can aggregate without re-parsing.

Both receive identical records. The JSONL is written first and flushed per
record, so an interrupted run still leaves a readable trail.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from app import db
from app.models import DecisionRecord

DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[1] / "data" / "runs"


class AuditLog:
    """Dual-sink writer for DecisionRecords."""

    def __init__(
        self,
        run_id: str,
        runs_dir: Path | str = DEFAULT_RUNS_DIR,
        db_path: Path | str | None = None,
        enable_sqlite: bool = True,
    ) -> None:
        self.run_id = run_id
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.runs_dir / f"{run_id}.jsonl"
        self._buffer: list[DecisionRecord] = []
        self._enable_sqlite = enable_sqlite
        self._db_path = db_path if db_path is not None else db.DEFAULT_DB_PATH
        self._fh = self.jsonl_path.open("w", encoding="utf-8")

    def write(self, record: DecisionRecord) -> None:
        self._fh.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        self._fh.flush()
        self._buffer.append(record)

    def write_many(self, records: Iterable[DecisionRecord]) -> None:
        for r in records:
            self.write(r)

    @property
    def records(self) -> list[DecisionRecord]:
        return list(self._buffer)

    def close(self, run_meta: dict | None = None) -> None:
        self._fh.close()
        if not self._enable_sqlite:
            return
        conn = db.connect(self._db_path)
        try:
            db.init_schema(conn)
            if run_meta is not None:
                db.insert_run(conn, run_meta)
            db.insert_decisions(conn, self._buffer)
        finally:
            conn.close()


def read_jsonl(path: Path | str) -> list[dict]:
    """Read back an audit trail. Used by evaluate.py and selfcheck.py."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
