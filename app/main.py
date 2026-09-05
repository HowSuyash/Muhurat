"""FastAPI application stub.

Day 1 exposes only a health check. The pipeline runs through scripts/, because
nothing about generating a corpus and measuring two arms needs an HTTP server.
The app exists so Day 2 can hang routes off the same settings and service layer
without restructuring anything.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.diagnosis.classifier import RulesClassifier
from app.settings import get_settings

app = FastAPI(
    title="Muhurat",
    description="Recovery is a timing problem. Razorpay AI Buildathon, Track 03.",
    version="0.1.0",
)


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
    }
