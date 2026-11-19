"""Embedders for the policy index.

- LiteLLMEmbedder: any LiteLLM embedding model, chosen by config only (default Gemini), the same
  provider-agnostic rule as the M1 chat client. Vectors are cached on disk by (model, size, text), so a
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
    name: str  # stored with the index, so vectors from different embedders are never mixed
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _normalise(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"local-hashing-{dim}"

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        tokens = _TOKEN.findall(text.lower())
        # words, plus each part of a hyphenated code (GAST-COLO -> gast, colo) so partial codes still match
        for tok in tokens + [p for t in tokens if "-" in t for p in t.split("-")]:
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalise([math.copysign(math.log1p(abs(x)), x) for x in v])

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class EmbeddingError(RuntimeError):
    pass


class LiteLLMEmbedder:
    BATCH = 50
    WINDOW_S = 60.0

    def __init__(self, settings: Settings | None = None, cache_path: Path | str | None = None, retries: int = 4):
        self.settings = settings or get_settings()
        self.texts_per_minute = self.settings.careflow_embedding_texts_per_minute
        self._sent: list[tuple[float, int]] = []  # (time, texts) sent in the current window
        self.model = self.settings.careflow_embedding_model
        self.dim = self.settings.careflow_embedding_dimensions
        self.name = f"{self.model}@{self.dim}"
        self.retries = retries
        self.calls = 0  # API requests made (for the build report)
        path = str(cache_path or EMBED_CACHE)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB NOT NULL)")

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.name}\n{text}".encode()).hexdigest()

    def _request(self, batch: list[str]) -> list[list[float]]:
        import litellm  # imported here so offline users of this module never need the network stack

        delay = 2.0
        for attempt in range(self.retries + 1):
            self._pace(len(batch))
            try:
                self.calls += 1
                resp = litellm.embedding(
                    model=self.model,
                    input=batch,
                    dimensions=self.dim,
                    timeout=self.settings.request_timeout_s,
                    **self.settings.provider_kwargs(self.model),
                )
                return [list(d["embedding"]) for d in resp.data]
            except (litellm.RateLimitError, litellm.ServiceUnavailableError, litellm.Timeout,
                    litellm.APIConnectionError) as exc:
                if attempt == self.retries:
                    raise EmbeddingError(f"{self.model}: {type(exc).__name__} after {attempt + 1} tries") from exc
                # A per-minute quota resets on a 60 s window, so short backoffs only burn the retries.
                self._sleep(self.WINDOW_S + 2 if isinstance(exc, litellm.RateLimitError) else delay)
                delay *= 2
        raise EmbeddingError("unreachable")

    def _pace(self, n_texts: int) -> None:
        """Keep under the provider's per-minute limit. Gemini counts every text in a batch as one request
        (free tier: 100 per minute), so pacing is by texts, not by HTTP calls."""
        now = time.monotonic()
        self._sent = [(t, n) for t, n in self._sent if now - t < self.WINDOW_S]
        used = sum(n for _, n in self._sent)
        if self._sent and used + n_texts > self.texts_per_minute:
            self._sleep(self.WINDOW_S - (now - self._sent[0][0]) + 1)
            self._sent = []
        self._sent.append((time.monotonic(), n_texts))

    def _sleep(self, seconds: float) -> None:
        time.sleep(max(0.0, seconds))

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        with self._lock:
            cached = {}
            for i in range(0, len(keys), 500):
                part = keys[i : i + 500]
                q = f"SELECT key, vec FROM vectors WHERE key IN ({','.join('?' * len(part))})"
                cached.update({k: v for k, v in self._db.execute(q, part)})
        missing = [i for i, k in enumerate(keys) if k not in cached]
        for start in range(0, len(missing), self.BATCH):
            idx = missing[start : start + self.BATCH]
            vectors = self._request([texts[i] for i in idx])
            if len(vectors) != len(idx) or any(len(v) != self.dim for v in vectors):
                raise EmbeddingError(f"{self.model} returned an unexpected shape")
            with self._lock:
                for i, v in zip(idx, vectors):
                    blob = struct.pack(f"{self.dim}f", *_normalise(v))
                    self._db.execute("INSERT OR REPLACE INTO vectors VALUES (?, ?)", (keys[i], blob))
                    cached[keys[i]] = blob
                self._db.commit()
        return [list(struct.unpack(f"{self.dim}f", cached[k])) for k in keys]

    def close(self) -> None:
        self._db.close()


def default_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    if settings.careflow_embedding_model == "local":
        return HashingEmbedder()
    return LiteLLMEmbedder(settings)
