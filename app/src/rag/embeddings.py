"""Embedders for the policy index.

- LiteLLMEmbedder: any LiteLLM embedding model, chosen by config only (default Gemini), the same
  provider-agnostic rule as the chat client. Vectors are cached on disk by (model, size, text), so a
  rebuild or a repeated query never pays twice.
- HashingEmbedder: a deterministic, offline bag-of-words embedder. Used by the tests and when
  CAREFLOW_EMBEDDING_MODEL=local. It matches words, not meaning, so it is a fallback and not the
  real index. (Local neural embedders such as fastembed need onnxruntime, which has no build for the
  project's Python 3.15.)
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Protocol

from ..config import EMBED_CACHE, Settings, get_settings


class Embedder(Protocol):
    """What the index needs from an embedder: one unit-length vector of `dimensions` floats per text."""

    name: str  # stored with the index, so vectors from different embedders are never mixed
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _normalise(vector: list[float]) -> list[float]:
    """Scale to unit length (a zero vector is returned unchanged), so cosine similarity is a dot product."""
    length = math.sqrt(sum(component * component for component in vector)) or 1.0
    return [component / length for component in vector]


_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")  # a lower-case word or number; a hyphenated code stays whole


class HashingEmbedder:
    """The offline fallback: signed feature hashing of the text's words into `dimensions` buckets, log-damped."""

    def __init__(self, dimensions: int = 512):
        self.dimensions = dimensions
        self.name = f"local-hashing-{dimensions}"

    def _build_vector(self, text: str) -> list[float]:
        buckets = [0.0] * self.dimensions
        tokens = _TOKEN.findall(text.lower())
        # words, plus each part of a hyphenated code (GAST-COLO -> gast, colo) so partial codes still match
        for token in tokens + [part for code in tokens if "-" in code for part in code.split("-")]:
            hashed = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
            buckets[hashed % self.dimensions] += 1.0 if (hashed >> 63) & 1 else -1.0  # the hash's top bit picks the sign
        return _normalise([math.copysign(math.log1p(abs(count)), count) for count in buckets])

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._build_vector(text) for text in texts]


class EmbeddingError(RuntimeError):
    """The provider could not be reached after every retry, or returned vectors of the wrong shape."""


_CACHE_LOOKUP_KEYS = 500  # keys per cache SELECT, under SQLite's limit on bound parameters (999 in old builds)


class LiteLLMEmbedder:
    """Embeds through LiteLLM with an on-disk vector cache, per-minute pacing and retries. Thread-safe."""

    BATCH = 50  # texts per API request
    WINDOW_S = 60.0  # length of the provider's rate-limit window

    def __init__(self, settings: Settings | None = None, cache_path: Path | str | None = None, retries: int = 4):
        self.settings = settings or get_settings()
        self.texts_per_minute = self.settings.careflow_embedding_texts_per_minute
        self._sent: list[tuple[float, int]] = []  # (time, texts) sent in the current window
        self.model = self.settings.careflow_embedding_model
        self.dimensions = self.settings.careflow_embedding_dimensions
        self.name = f"{self.model}@{self.dimensions}"
        self.retries = retries
        self.calls = 0  # API requests made (for the build report)
        path = str(cache_path or EMBED_CACHE)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # the cache database
        self._pace_lock = threading.Lock()  # the rate-limit window and the call count
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB NOT NULL)")

    def _make_key(self, text: str) -> str:
        """Cache key: the model and vector size are part of it, so a change of either never reuses a vector."""
        return hashlib.sha256(f"{self.name}\n{text}".encode()).hexdigest()

    def _request(self, batch: list[str]) -> list[list[float]]:
        """One embedding request, retried on transient provider errors; EmbeddingError when retries run out."""
        import litellm  # imported here so offline users of this module never need the network stack

        delay = 2.0  # seconds before the first retry of a non-quota error; doubles each attempt
        for attempt in range(self.retries + 1):
            self._pace(len(batch))
            try:
                with self._pace_lock:
                    self.calls += 1
                response = litellm.embedding(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions,
                    timeout=self.settings.request_timeout_s,
                    **self.settings.get_provider_options(self.model),
                )
                return [list(item["embedding"]) for item in response.data]
            except (litellm.RateLimitError, litellm.ServiceUnavailableError, litellm.Timeout,
                    litellm.APIConnectionError) as error:
                if attempt == self.retries:
                    raise EmbeddingError(f"{self.model}: {type(error).__name__} after {attempt + 1} tries") from error
                # A per-minute quota resets on a 60 s window, so short backoffs only burn the retries.
                self._sleep(self.WINDOW_S + 2 if isinstance(error, litellm.RateLimitError) else delay)
                delay *= 2
        raise EmbeddingError("unreachable")

    def _pace(self, text_count: int) -> None:
        """Keep under the provider's per-minute limit. Gemini counts every text in a batch as one request
        (free tier: 100 per minute), so pacing is by texts, not by HTTP calls. Held under a lock, so two
        threads cannot both decide there is room in the same window."""
        with self._pace_lock:
            now = time.monotonic()
            self._sent = [(sent_at, count) for sent_at, count in self._sent if now - sent_at < self.WINDOW_S]
            used = sum(count for _, count in self._sent)
            if self._sent and used + text_count > self.texts_per_minute:
                self._sleep(self.WINDOW_S - (now - self._sent[0][0]) + 1)
                self._sent = []
            self._sent.append((time.monotonic(), text_count))

    def _sleep(self, seconds: float) -> None:  # a method so tests can replace it
        time.sleep(max(0.0, seconds))

    def _load_cached(self, keys: list[str]) -> dict[str, bytes]:
        """{key: packed vector} for the keys already in the cache."""
        cached: dict[str, bytes] = {}
        with self._lock:
            for start in range(0, len(keys), _CACHE_LOOKUP_KEYS):
                part = keys[start : start + _CACHE_LOOKUP_KEYS]
                statement = f"SELECT key, vec FROM vectors WHERE key IN ({','.join('?' * len(part))})"
                cached.update(self._db.execute(statement, part))
        return cached

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Unit-length vectors, in the order of `texts`. Only texts missing from the cache are sent to the
        provider; vectors are stored as packed float32."""
        keys = [self._make_key(text) for text in texts]
        cached = self._load_cached(keys)
        missing = [index for index, key in enumerate(keys) if key not in cached]
        for start in range(0, len(missing), self.BATCH):
            positions = missing[start : start + self.BATCH]
            vectors = self._request([texts[index] for index in positions])
            if len(vectors) != len(positions) or any(len(vector) != self.dimensions for vector in vectors):
                raise EmbeddingError(f"{self.model} returned an unexpected shape")
            with self._lock:
                for index, vector in zip(positions, vectors, strict=True):
                    blob = struct.pack(f"{self.dimensions}f", *_normalise(vector))
                    self._db.execute("INSERT OR REPLACE INTO vectors VALUES (?, ?)", (keys[index], blob))
                    cached[keys[index]] = blob
                self._db.commit()
        return [list(struct.unpack(f"{self.dimensions}f", cached[key])) for key in keys]

    def close(self) -> None:
        self._db.close()


def build_embedder(settings: Settings | None = None) -> Embedder:
    """The embedder the settings ask for: the offline one for CAREFLOW_EMBEDDING_MODEL=local, else LiteLLM."""
    settings = settings or get_settings()
    if settings.careflow_embedding_model == "local":
        return HashingEmbedder()
    return LiteLLMEmbedder(settings)
