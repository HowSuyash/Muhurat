"""FastAPI app: the results dashboard and its data endpoint.

The dashboard reads whatever the last `python -m scripts.evaluate` produced. It
is deliberately thin — a viewer over `data/runs/summary.json`, not a second
source of truth. Every number it shows came out of the benchmark, so there is no
path by which the page and the CLI can disagree.

Notably it does NOT import ground truth. The ceiling is computed inside
`scripts/evaluate.py` (which L5 whitelists) and written into the summary, so
serving this page adds no new reader of the hidden windows.

    uvicorn app.main:app --reload   ->  http://127.0.0.1:8000
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from app.diagnosis.classifier import RulesClassifier
from app.settings import get_settings

STATIC_DIR = Path(__file__).resolve().parent / "static"
RUNS_DIR = Path(__file__).resolve().parents[1] / "data" / "runs"
SUMMARY_PATH = RUNS_DIR / "summary.json"
EXPLORER_PATH = RUNS_DIR / "explorer.json"
REPLAY_PATH = RUNS_DIR / "replay.json"

app = FastAPI(
    title="Muhurat",
    description="Recovery is a timing problem. Razorpay AI Buildathon, Track 03.",
    version="0.1.0",
)


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    """The results dashboard."""
    return FileResponse(STATIC_DIR / "dashboard.html", media_type="text/html")


@app.get("/api/summary")
def summary() -> JSONResponse:
    """The latest benchmark results, exactly as evaluate.py wrote them."""
    if not SUMMARY_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No results yet. Run: python -m scripts.generate_corpus && "
                "python -m scripts.run_baseline && python -m scripts.evaluate"
            ),
        )
    return JSONResponse(json.loads(SUMMARY_PATH.read_text(encoding="utf-8")))


@app.get("/api/explorer")
def explorer() -> JSONResponse:
    """Per-payment detail: what every arm did, and where the window actually was.

    Produced by scripts/export_explorer.py after a run. Like the summary, it is a
    post-hoc view -- no policy sees any of this at decision time.
    """
    if not EXPLORER_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No explorer data yet. Run: python -m scripts.run_baseline && "
                "python -m scripts.evaluate && python -m scripts.export_explorer"
            ),
        )
    return JSONResponse(json.loads(EXPLORER_PATH.read_text(encoding="utf-8")))


@app.get("/api/replay")
def replay() -> JSONResponse:
    """Compact event stream the dashboard replays: attempts, recoveries, windows."""
    if not REPLAY_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No replay data yet. Run: python -m scripts.export_replay",
        )
    return JSONResponse(json.loads(REPLAY_PATH.read_text(encoding="utf-8")))


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    classifier = RulesClassifier()
    return {
        "status": "ok",
        # Presence only. The credential values are never read or returned.
        "razorpay_credentials_configured": settings.credentials_configured,
        "rules_loaded": len(classifier.table),
        "classifier_version": classifier.version,
        "results_available": SUMMARY_PATH.exists(),
        "explorer_available": EXPLORER_PATH.exists(),
        "replay_available": REPLAY_PATH.exists(),
    }
