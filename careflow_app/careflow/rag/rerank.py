"""Rerankers: score (question, passage) pairs together, after hybrid retrieval has narrowed the corpus.

A bi-encoder (the dense index) embeds the question and the passage separately; a cross-encoder reads
them together, so it can tell "Priority referral: 2 business days" from "Routine referral: 5 business
days" for a question about Priority referrals. It is too slow to run over the whole corpus, so it runs
on the ~20 fused candidates only.

Scores are probabilities in [0, 1] (sigmoid of the model's logit). That is what makes one abstain
threshold meaningful: below it, the clinic's documents do not answer the question (retriever.py).

- CrossEncoderReranker: a local sentence-transformers cross-encoder on CPU (default
  cross-encoder/ms-marco-MiniLM-L-12-v2, ~130 MB, downloaded once into the Hugging Face cache;
  chosen over L-6 for a ~100x gap between answerable and unanswerable scores, see run_m4_eval.py). No API
  quota per query. Loaded lazily, so importing this module costs nothing.
- Tests pass their own Reranker; nothing here needs the network to import.
"""

from __future__ import annotations

import math
import threading
from typing import Protocol


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_length: int = 512, device: str = "cpu", revision: str | None = None):
        self.name = model_name
        self.revision = revision  # pin a model commit for reproducible scores (None = latest)
        self.max_length = max_length
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder  # heavy (torch): only when reranking runs

                kwargs = {"max_length": self.max_length, "device": self.device, "revision": self.revision}
                try:
                    # The local cache first: no Hub request on a warm start, and it works offline or in a
                    # container with the weights baked in (feedback M4-7).
                    self._model = CrossEncoder(self.name, local_files_only=True, **kwargs)
                except (OSError, ValueError):  # not cached yet (huggingface raises an OSError subclass)
                    self._model = CrossEncoder(self.name, **kwargs)  # first run: download once
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        import torch

        model = self._load()
        # Raw logits, then our own sigmoid: the probability scale does not depend on the library's default.
        logits = model.predict([(query, p) for p in passages], activation_fn=torch.nn.Identity(),
                               show_progress_bar=False, convert_to_numpy=True)
        return [_sigmoid(float(x)) for x in logits]


def default_reranker(model_name: str) -> Reranker | None:
    """None when reranking is switched off (CAREFLOW_RERANKER_MODEL=none)."""
    if not model_name or model_name.lower() == "none":
        return None
    return CrossEncoderReranker(model_name)
