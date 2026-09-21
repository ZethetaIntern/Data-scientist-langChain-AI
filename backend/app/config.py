"""Application settings.

Everything is configured through environment variables (or a local ``.env`` file).
All variables are prefixed with ``DS_`` except the provider API keys, which use the
standard names that the LangChain integrations already understand
(``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``GROQ_API_KEY``, ``OLLAMA_BASE_URL``).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["auto", "openai", "anthropic", "groq", "ollama", "heuristic"]

DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.1",
}

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration, loaded once at start-up."""

    model_config = SettingsConfigDict(
        env_prefix="DS_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: Provider = "auto"
    llm_model: str | None = None
    llm_temperature: float = 0.0
    max_iterations: int = Field(default=8, ge=1, le=25)
    enable_code_tool: bool = False

    max_upload_mb: int = Field(default=25, ge=1, le=500)
    max_rows: int = Field(default=200_000, ge=100)
    max_columns: int = Field(default=100, ge=2)
    max_datasets: int = Field(default=6, ge=1, le=50)

    artifact_dir: Path = REPO_ROOT / "artifacts"
    cors_origins: str = "http://localhost:5173"

    # ── derived helpers ────────────────────────────────────────────────────
    def resolved_provider(self) -> str:
        """Turn ``auto`` into a concrete provider based on the keys that are present."""
        if self.llm_provider != "auto":
            return self.llm_provider
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.environ.get("GROQ_API_KEY"):
            return "groq"
        return "heuristic"

    def resolved_model(self) -> str | None:
        provider = self.resolved_provider()
        if provider == "heuristic":
            return None
        return self.llm_model or DEFAULT_MODELS.get(provider)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    return settings
