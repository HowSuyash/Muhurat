"""Configuration, loaded from .env via python-dotenv.

The Razorpay credentials are declared here but read by nothing in the Day-1
pipeline -- the simulated executor has no use for them. They exist so that
Day 2's real `RazorpayExecutor` has a typed, single place to get them from.

They are `SecretStr`, so printing a Settings object, logging it, or dumping it
into an error message yields `**********` rather than the key. `credentials_
configured` reports presence only, and is the only thing that ever gets logged.

Note: this module is the one part of the project that needs third-party
packages. The measurement pipeline (corpus -> diagnosis -> execution -> eval)
is pure stdlib, so a broken install can never block the numbers.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

# Explicit dotenv load so the values are present regardless of how the process
# was started (uvicorn, python -m, or an IDE runner).
load_dotenv(ENV_PATH, override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    razorpay_key_id: SecretStr = SecretStr("")
    razorpay_key_secret: SecretStr = SecretStr("")

    # Paths, so nothing hardcodes a location.
    project_root: Path = PROJECT_ROOT
    rules_path: Path = PROJECT_ROOT / "config" / "rules.toml"
    outcome_model_path: Path = PROJECT_ROOT / "config" / "outcome_model.toml"
    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"
    runs_dir: Path = PROJECT_ROOT / "data" / "runs"
    db_path: Path = PROJECT_ROOT / "data" / "recovery.db"

    default_seed: int = 42
    default_corpus_size: int = 300

    @property
    def credentials_configured(self) -> bool:
        """Presence check only. Never exposes, compares, or logs the values."""
        return bool(
            self.razorpay_key_id.get_secret_value()
            and self.razorpay_key_secret.get_secret_value()
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
