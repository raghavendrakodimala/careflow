"""Settings. Keys come from the toolkit's .env (never printed); the model is chosen here only."""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]  # this file is <repo>/app/src/config.py
TOOLKIT_DIR = REPO_ROOT / "capstone-data-toolkit"
DATA_DIR = TOOLKIT_DIR / "data" / "careflow"
# Runtime SQLite database (bookings, flags), seeded from DATA_DIR's CSVs. Gitignored via out/.
# The seeded DATA_DIR is never written. Each path below can be moved with its CAREFLOW_* environment variable.
OUT_DIR = REPO_ROOT / "app" / "out"
RUNTIME_DB = Path(os.environ.get("CAREFLOW_RUNTIME_DB") or OUT_DIR / "runtime.db")
# Session checkpoints (LangGraph SqliteSaver), the local Qdrant policy index, and an embedding cache
# so rebuilding the index never pays for the same text twice. All gitignored via out/.
SESSIONS_DB = Path(os.environ.get("CAREFLOW_SESSIONS_DB") or OUT_DIR / "sessions.db")
QDRANT_PATH = Path(os.environ.get("CAREFLOW_QDRANT_PATH") or OUT_DIR / "qdrant")
EMBED_CACHE = Path(os.environ.get("CAREFLOW_EMBED_CACHE") or OUT_DIR / "embeddings.sqlite")


# The mode settings. Each member equals its string value; the settings hold the plain string.
class TracingMode(StrEnum):
    """CAREFLOW_TRACING: where spans go."""

    OFF = "off"
    JSONL = "jsonl"
    LANGFUSE = "langfuse"


class GuardClassifierMode(StrEnum):
    """CAREFLOW_GUARD_CLASSIFIER: the input gate's classifier."""

    NONE = "none"      # rules only
    LOCAL = "local"    # in-process, fast
    HOSTED = "hosted"  # the chat model


class PolicySource(StrEnum):
    """CAREFLOW_POLICY_SOURCE: the copy of the corpus the policy index is built from."""

    PDF = "pdf"
    MARKDOWN = "markdown"


# Allowed values of the mode settings. The first is the one an empty value means.
_MODES = {
    "careflow_tracing": tuple(mode.value for mode in TracingMode),
    "careflow_guard_classifier": tuple(mode.value for mode in GuardClassifierMode),
    "careflow_policy_source": tuple(source.value for source in PolicySource),
}

# Defaults that a setting below shares with the class it configures (reliability.py, guardrails.py).
DEFAULT_TOOL_ATTEMPTS = 3          # tries per retryable tool call, the first included
DEFAULT_BREAKER_FAILURES = 3       # failures in a row that open a circuit breaker
DEFAULT_BREAKER_COOLDOWN_S = 30.0  # seconds an open breaker waits before allowing a trial call
GUARD_TIME_LIMIT_S = 2.0           # OP-CS-014 9.3: seconds the input-gate classifier has to answer
# CPU threads for the local models (classifier encoder and reranker). PyTorch's default is one per core;
# on a many-core machine that gives the classifier a long latency tail when anything else runs (measured
# on 22 threads: slowest gate call 216 ms idle and past the 2-second limit under load, against 34 ms with
# 4 threads). The cost is about 0.7 s more per reranked search (2.6 s -> 3.3 s); 8 threads was no faster.
LOCAL_MODEL_THREADS = 4


def limit_local_model_threads() -> None:
    """Cap PyTorch's CPU threads before a local model runs. Process-wide; never raises the current number."""
    import torch  # imported here: only processes that run a local model pay for it

    torch.set_num_threads(min(LOCAL_MODEL_THREADS, torch.get_num_threads()))


class Settings(BaseSettings):
    """Every field can be set by an environment variable of the same name (or its alias), any case, or in
    one of the .env files below; the app's own .env wins over the toolkit's."""

    model_config = SettingsConfigDict(
        env_file=(TOOLKIT_DIR / ".env", REPO_ROOT / "app" / ".env"),
        extra="ignore",
        populate_by_name=True,
    )

    # LiteLLM model string. Swap provider by changing this only, e.g.
    # "openrouter/meta-llama/llama-3.3-70b-instruct" or "ollama/llama3.1:8b".
    # The lite model is first because the free tier allows gemini-flash-latest only 20 requests a day.
    careflow_model: str = "gemini/gemini-flash-lite-latest"
    # Comma-separated, tried in order if the primary keeps failing (503, timeout).
    careflow_fallback_models: str = "gemini/gemini-flash-latest"

    # Keys are SecretStr: masked in repr(), str() and dumps of the settings; read with .get_secret_value().
    gemini_api_key: SecretStr = Field(
        SecretStr(""), validation_alias=AliasChoices("DATAGEN_GEMINI_API_KEY", "GEMINI_API_KEY"))
    openrouter_api_key: SecretStr = Field(
        SecretStr(""), validation_alias=AliasChoices("DATAGEN_OPENROUTER_API_KEY", "OPENROUTER_API_KEY")
    )
    ollama_host: str = Field("http://localhost:11434", validation_alias="DATAGEN_OLLAMA_HOST")

    # Policy index. Any LiteLLM embedding model; "local" uses the offline hashing embedder (no API).
    careflow_embedding_model: str = "gemini/gemini-embedding-001"
    careflow_embedding_dimensions: int = 768
    # Gemini's free tier counts each embedded text as a request: 100 per minute. Stay under it.
    careflow_embedding_texts_per_minute: int = 90

    # Retrieval. The index is built from the PDF copy of the corpus ("the PDF is what production looks
    # like", toolkit README); "markdown" is the clean copy, used as the parity reference.
    careflow_policy_source: str = PolicySource.PDF.value
    # Local cross-encoder (sentence-transformers) that reranks the fused candidates; "none" switches it off.
    careflow_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    # Abstain when the best reranked passage scores below this (probability, 0..1). Calibrated by
    # scripts/run_retrieval_eval.py on answerable vs unanswerable questions: with the offline hashing embedder the
    # lowest answerable score was 0.314 and the highest unanswerable 0.003.
    careflow_rerank_min_score: float = 0.03
    careflow_retrieval_candidates: int = 20  # passages fetched from the index before reranking
    # A Qdrant server URL, for when several processes share the index; empty = local mode at out/qdrant.
    careflow_qdrant_url: str = ""
    # Tracing (off | jsonl | langfuse). LangFuse needs both keys; without them the tracer falls back to the
    # local JSONL file.
    careflow_tracing: str = TracingMode.JSONL.value
    langfuse_public_key: SecretStr = SecretStr("")
    langfuse_secret_key: SecretStr = SecretStr("")
    langfuse_host: str = ""
    # Reliability: attempts per retryable tool call, and a circuit breaker per backend that opens after
    # this many failures and allows a trial call once the cool-down has passed.
    careflow_tool_retries: int = DEFAULT_TOOL_ATTEMPTS
    careflow_breaker_failures: int = DEFAULT_BREAKER_FAILURES
    careflow_breaker_cooldown_s: float = DEFAULT_BREAKER_COOLDOWN_S
    # Input-gate classifier time limit (OP-CS-014 9.3 says 2 seconds), and the API.
    careflow_guard_timeout_s: float = GUARD_TIME_LIMIT_S
    careflow_guard_classifier: str = GuardClassifierMode.LOCAL.value  # local (in-process, fast) | hosted (the chat model) | none (rules only)
    careflow_api_key: SecretStr = SecretStr("")  # if set, every /v1 request must send it as X-API-Key
    careflow_api_use_mcp: bool = True  # the API reaches the EHR tools through the MCP server

    # Per model call: LiteLLM's timeout, and its own retries on transient errors before a fallback model is tried.
    request_timeout_s: int = 90
    transport_retries: int = 3

    @field_validator("careflow_tracing", "careflow_guard_classifier", "careflow_policy_source")
    @classmethod
    def _validate_mode(cls, value: str, info) -> str:
        """A mistyped mode stops the application at start-up with the allowed values, where it would
        otherwise fall back silently (for example to no tracing). Case and surrounding spaces are ignored;
        an empty value means the setting's "off" choice."""
        allowed = _MODES[info.field_name]
        mode = (value or "").strip().lower() or allowed[0]
        if mode not in allowed:
            raise ValueError(f"must be one of {', '.join(allowed)} (got {value!r})")
        return mode

    @property
    def models(self) -> list[str]:
        """The primary model, then the fallbacks in order, without repeats of the primary."""
        extra = [model.strip() for model in self.careflow_fallback_models.split(",") if model.strip()]
        return [self.careflow_model, *[model for model in extra if model != self.careflow_model]]

    def get_provider_options(self, model: str) -> dict:
        """Credentials for a model string, chosen by its provider prefix."""
        if model.startswith("gemini/"):
            return {"api_key": self.gemini_api_key.get_secret_value()}
        if model.startswith("openrouter/"):
            return {"api_key": self.openrouter_api_key.get_secret_value()}
        if model.startswith("ollama"):
            return {"api_base": self.ollama_host}
        return {}


def get_settings() -> Settings:
    """Fresh settings on every call (not cached), so a changed environment is picked up."""
    return Settings()
