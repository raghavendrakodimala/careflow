"""The local Clinical Content Classifier for the input gate (OP-CS-014 9.2, 9.3).

The policy wants every message scored by a classifier, blocked at 0.30 or above, and treated as clinical
if the classifier takes more than 2 seconds. A hosted model cannot promise 2 seconds (measured: never
under 2.2 s, so 19 of 26 messages were refused unseen). This classifier runs in-process:

    message -> sentences -> one embedding each (all-MiniLM-L6-v2, CPU) -> logistic regression -> highest score

Scoring sentences, not the whole message, is what makes it usable: "I need a dermatology appointment.
Is this mole something to worry about?" is one clinical sentence inside a booking request, and a
whole-message embedding blurs the two. Warm, a decision takes tens of milliseconds.

The weights (src/data/clinical_gate.npz, 384 numbers and an intercept) are produced offline by
scripts/train_guard_classifier.py, which also prints the cross-validation figures. Inference here is
numpy only. Any failure to load or score raises, and guardrails.check_input turns that into a block.

Known limit, measured: it over-blocks messages that describe symptoms without asking for advice. That
is why it guards the chat path, and the triage workflow keeps its rules plus the intake parse.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .config import GuardClassifierMode, Settings, get_settings
from .guardrails import THRESHOLD, Classifier, ClinicalRisk, GuardCategory, build_hosted_classifier

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # default; _load() takes the name stored with the weights
WEIGHTS = Path(__file__).parent / "data" / "clinical_gate.npz"  # arrays: w (384,), b (scalar), model (name)
MIN_UNIT_WORDS = 4  # a shorter fragment ("Thanks.", "Is it?") says too little to be scored alone
_MAX_TOKENS = 128  # tokens of a sentence the embedding model reads; the rest is truncated
_ENCODE_BATCH = 64  # sentences per forward pass
_SPLIT = re.compile(r"(?<=[.?!])\s+|\n+")  # sentence boundary: whitespace after . ? or !, or a line break


def split_sentences(text: str) -> list[str]:
    """The units that are scored. Fragments under MIN_UNIT_WORDS words join the unit before them. Never
    empty: a blank message gives one placeholder unit. Every sentence is kept, however long the message:
    dropping the tail would let a question placed after harmless text go unchecked. The gate's time limit
    is the bound on work (about 1 second for the longest message the API accepts)."""
    units: list[str] = []
    for part in (fragment.strip() for fragment in _SPLIT.split(text or "")):
        if not part:
            continue
        if units and len(part.split()) < MIN_UNIT_WORDS:
            units[-1] += " " + part
        else:
            units.append(part)
    return units or [(text or "").strip() or "."]


Encoder = Callable[[list[str]], Any]  # sentences -> (n, d) array of unit-length embeddings


class LocalClinicalClassifier:
    """A guardrails.Classifier: call it with a message to get the highest clinical score of its sentences."""

    def __init__(self, weights_path: Path | str | None = None, encoder: Encoder | None = None):
        self.weights_path = Path(weights_path or WEIGHTS)
        self._encoder = encoder  # injected in tests; otherwise the sentence-transformers model, loaded lazily
        self._weights: tuple[np.ndarray, float] | None = None  # (weights, bias) of the logistic regression
        self.model_name = MODEL_NAME
        self._lock = threading.Lock()

    def _load(self) -> None:
        """Load the weights and, unless an encoder was injected, the embedding model. Runs once; the lock
        stops two requests loading at the same time."""
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
                model.max_seq_length = _MAX_TOKENS
                self._encoder = lambda units: model.encode(
                    units, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True,
                    batch_size=_ENCODE_BATCH)

    def score_sentences(self, text: str) -> list[tuple[str, float]]:
        """(sentence, score) for each unit of the message."""
        if self._weights is None or self._encoder is None:
            self._load()
        units = split_sentences(text)
        weights, bias = self._weights
        embeddings = np.asarray(self._encoder(units), dtype=np.float32)
        probabilities = 1.0 / (1.0 + np.exp(-(embeddings @ weights + bias)))  # logistic regression
        return list(zip(units, (float(probability) for probability in probabilities), strict=True))

    def __call__(self, text: str) -> ClinicalRisk:
        """The message's risk: its most clinical sentence decides. Never category "injection" (the rules
        catch those); the gate applies its own threshold to the score."""
        score = max(probability for _, probability in self.score_sentences(text))
        category = GuardCategory.CLINICAL if score >= THRESHOLD else GuardCategory.NONE
        return ClinicalRisk(score=min(1.0, max(0.0, score)), category=category.value)

    def warm(self) -> None:
        """Load the model and run one message, so the first real message is not the slow one."""
        self("Can I book an appointment for next week?")


def build_classifier(settings: Settings | None = None) -> Classifier | None:
    """The input-gate classifier the settings ask for: local (default), hosted (the chat model) or none."""
    settings = settings or get_settings()
    mode = (settings.careflow_guard_classifier or GuardClassifierMode.NONE).lower()
    if mode == GuardClassifierMode.LOCAL:
        return LocalClinicalClassifier()
    if mode == GuardClassifierMode.HOSTED:
        return build_hosted_classifier(settings)
    return None
