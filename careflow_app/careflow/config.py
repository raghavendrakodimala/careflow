"""Settings. Keys come from the toolkit's .env (never printed); the model is chosen here only."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLKIT_DIR = REPO_ROOT / "capstone-data-toolkit"
DATA_DIR = TOOLKIT_DIR / "data" / "careflow"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(TOOLKIT_DIR / ".env", REPO_ROOT / "careflow_app" / ".env"),
        extra="ignore",
        populate_by_name=True,
    )

    # LiteLLM model string. Swap provider by changing this only, e.g.
    # "openrouter/meta-llama/llama-3.3-70b-instruct" or "ollama/llama3.1:8b".
    careflow_model: str = "gemini/gemini-flash-latest"
    # Comma-separated, tried in order if the primary keeps failing (503, timeout).
    careflow_fallback_models: str = "gemini/gemini-flash-lite-latest"

    gemini_api_key: str = Field("", validation_alias=AliasChoices("DATAGEN_GEMINI_API_KEY", "GEMINI_API_KEY"))
    openrouter_api_key: str = Field(
        "", validation_alias=AliasChoices("DATAGEN_OPENROUTER_API_KEY", "OPENROUTER_API_KEY")
    )
    ollama_host: str = Field("http://localhost:11434", validation_alias="DATAGEN_OLLAMA_HOST")

    request_timeout_s: int = 90
    transport_retries: int = 3

    @property
    def models(self) -> list[str]:
        extra = [m.strip() for m in self.careflow_fallback_models.split(",") if m.strip()]
        return [self.careflow_model, *[m for m in extra if m != self.careflow_model]]

    def provider_kwargs(self, model: str) -> dict:
        """Credentials for a model string, chosen by its provider prefix."""
        if model.startswith("gemini/"):
            return {"api_key": self.gemini_api_key}
        if model.startswith("openrouter/"):
            return {"api_key": self.openrouter_api_key}
        if model.startswith("ollama"):
            return {"api_base": self.ollama_host}
        return {}


def get_settings() -> Settings:
    return Settings()
