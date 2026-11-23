"""M8: the local Clinical Content Classifier for the input gate (OP-CS-014 9.2, 9.3).

The policy wants every message scored by a classifier, blocked at 0.30 or above, and treated as clinical
if the classifier takes more than 2 seconds. A hosted model cannot promise 2 seconds (the M8 live run:
never under 2.2 s, so 19 of 26 messages were refused unseen). This classifier runs in-process:

    message -> sentences -> one embedding each (all-MiniLM-L6-v2, CPU) -> logistic regression -> highest score

Scoring sentences, not the whole message, is what makes it usable: "I need a dermatology appointment.
Is this mole something to worry about?" is one clinical sentence inside a booking request, and a
whole-message embedding blurs the two. Warm, a decision takes tens of milliseconds.

The weights (careflow/data/clinical_gate.npz, 384 numbers and an intercept) are produced offline by
scripts/train_guard_classifier.py, which also prints the cross-validation figures. Inference here is
numpy only. Any failure to load or score raises, and guardrails.check_input turns that into a block.

Known limit, measured: it over-blocks messages that describe symptoms without asking for advice. That
is why it guards the chat path, and the triage workflow keeps its rules plus the intake parse.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import Settings, get_settings
from .guardrails import Classifier, ClinicalRisk, default_classifier

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
WEIGHTS = Path(__file__).parent / "data" / "clinical_gate.npz"
MAX_SENTENCES = 40  # a bound on work per message; later sentences of a very long message are dropped
_SPLIT = re.compile(r"(?<=[.?!])\s+|\n+")


def sentences(text: str) -> list[str]:
    """The units that are scored. Fragments under four words join the unit before them."""
    out: list[str] = []
    for part in (p.strip() for p in _SPLIT.split(text or "")):
        if not part:
            continue
        if out and len(part.split()) < 4:
            out[-1] += " " + part
        else:
            out.append(part)
    return (out or [(text or "").strip() or "."])[:MAX_SENTENCES]


Encoder = Callable[[list[str]], Any]  # sentences -> (n, d) array of unit-length embeddings


class LocalClinicalClassifier:
    def __init__(self, weights_path: Path | str | None = None, encoder: Encoder | None = None):
        self.weights_path = Path(weights_path or WEIGHTS)
        self._encoder = encoder  # injected in tests; otherwise the sentence-transformers model, loaded lazily
        self._weights: tuple[np.ndarray, float] | None = None
        self.model_name = MODEL_NAME
        self._lock = threading.Lock()

    def _load(self) -> None:
        with self._lock:
            if self._weights is None:
                data = np.load(self.weights_path, allow_pickle=False)
                self.model_name = str(data["model"])
                self._weights = (data["w"].astype(np.float32), float(data["b"]))
            if self._encoder is None:
                from sentence_transformers import SentenceTransformer

                try:  # the cache first: no network call on the request path
                    model = SentenceTransformer(self.model_name, device="cpu", local_files_only=True)
                except Exception:
                    model = SentenceTransformer(self.model_name, device="cpu")
                model.max_seq_length = 128
                self._encoder = lambda units: model.encode(
                    units, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True, batch_size=64)

    def scores(self, text: str) -> list[tuple[str, float]]:
        """(sentence, score) for each unit of the message."""
        if self._weights is None or self._encoder is None:
            self._load()
        units = sentences(text)
        w, b = self._weights
        probabilities = 1.0 / (1.0 + np.exp(-(np.asarray(self._encoder(units), dtype=np.float32) @ w + b)))
        return list(zip(units, (float(p) for p in probabilities)))

    def __call__(self, text: str) -> ClinicalRisk:
        score = max(p for _, p in self.scores(text))
        return ClinicalRisk(score=min(1.0, max(0.0, score)), category="clinical" if score >= 0.30 else "none")

    def warm(self) -> None:
        """Load the model and run one message, so the first real message is not the slow one."""
        self("Can I book an appointment for next week?")


def build_classifier(settings: Settings | None = None) -> Classifier | None:
    """The input-gate classifier the settings ask for: local (default), hosted (the chat model) or none."""
    settings = settings or get_settings()
    mode = (settings.careflow_guard_classifier or "none").lower()
    if mode == "local":
        return LocalClinicalClassifier()
    if mode == "hosted":
        return default_classifier(settings)
    return None
