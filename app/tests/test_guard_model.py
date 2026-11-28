"""The local Clinical Content Classifier. The first tests use a fake encoder and need no model. The
last group loads the real encoder from the local Hugging Face cache and is skipped where it is absent."""

import json
import time

import numpy as np
import pytest

from scripts.run_system_eval import load_cases
from src.config import DATA_DIR, LOCAL_MODEL_THREADS, Settings, limit_local_model_threads
from src.guard_model import MODEL_NAME, WEIGHTS, LocalClinicalClassifier, build_classifier, split_sentences
from src.guardrails import THRESHOLD, TIME_LIMIT_S, check_input, check_rules

from .test_guardrails import ALLOWED_MESSAGES


def _write_weights(tmp_path, weight_vector=(4.0, 0.0), bias=-2.0, model="fake-encoder"):
    path = tmp_path / "gate.npz"
    np.savez(path, w=np.array(weight_vector, dtype=np.float32), b=np.float32(bias), model=np.array(model))
    return path


def _make_encoder(table):
    """Sentences containing a key get that vector; everything else is the zero vector."""
    return lambda units: np.array([next((vector for key, vector in table.items() if key in unit), (0.0, 0.0)) for unit in units], dtype=np.float32)


# ---- sentence splitting ------------------------------------------------------------------------------
def test_sentences_are_split_and_short_fragments_join_the_one_before():
    assert split_sentences("I need an appointment for my knee. Do you think it is broken? Thanks.") == [
        "I need an appointment for my knee.", "Do you think it is broken? Thanks."]
    two_lines = "Hello,\n\nPlease book me in for next week."
    assert split_sentences(two_lines) == ["Hello,", "Please book me in for next week."]
    assert split_sentences("") == ["."] and len(split_sentences("One two three four. " * 100)) == 100


# ---- scoring with a fake encoder ---------------------------------------------------------------------
def test_message_score_is_its_highest_sentence(tmp_path):
    classifier = LocalClinicalClassifier(_write_weights(tmp_path), _make_encoder({"dose": (1.0, 0.0), "slot": (0.2, 0.0)}))
    scores = dict(classifier.score_sentences("Please find me a slot on Tuesday. What dose should I be on now?"))
    assert scores["What dose should I be on now?"] == pytest.approx(1 / (1 + np.exp(-2.0)))
    risk = classifier("Please find me a slot on Tuesday. What dose should I be on now?")
    assert risk.score == pytest.approx(0.8808, abs=1e-3) and risk.category == "clinical"
    assert classifier("Please find me a slot on Tuesday.").category == "none"
    assert classifier.model_name == "fake-encoder"


@pytest.mark.parametrize("logit, blocked", [(-1.0, False), (-0.86, False), (-0.84, True), (2.0, True)])
def test_the_policy_threshold_applies_to_the_score(tmp_path, logit, blocked):
    # sigmoid(-0.8473) = 0.30
    classifier = LocalClinicalClassifier(_write_weights(tmp_path, weight_vector=(1.0, 0.0), bias=0.0), _make_encoder({"x": (logit, 0.0)}))
    result = check_input("x marks the message here", classifier)
    assert result.blocked is blocked and result.layer == "classifier"
    assert (result.score >= THRESHOLD) is blocked


def test_missing_or_broken_weights_fail_closed(tmp_path):
    for classifier in (LocalClinicalClassifier(tmp_path / "absent.npz", _make_encoder({})),
                LocalClinicalClassifier(_write_weights(tmp_path), lambda units: (_ for _ in ()).throw(RuntimeError("encoder died")))):
        result = check_input("Can I book an appointment?", classifier)
        assert (result.blocked, result.layer) == (True, "fail_closed") and "9.3" in result.reason


def test_classifier_is_chosen_by_the_setting():
    assert isinstance(build_classifier(Settings(careflow_guard_classifier="local")), LocalClinicalClassifier)
    assert build_classifier(Settings(careflow_guard_classifier="none")) is None
    hosted = build_classifier(Settings(careflow_guard_classifier="hosted"))
    assert callable(hosted) and not isinstance(hosted, LocalClinicalClassifier)
    assert Settings().careflow_guard_classifier == "local" and Settings().careflow_guard_timeout_s == TIME_LIMIT_S


def test_shipped_weights_match_the_configured_encoder():
    data = np.load(WEIGHTS, allow_pickle=False)
    assert str(data["model"]) == MODEL_NAME and data["w"].shape == (384,) and np.isfinite(data["w"]).all()


# ---- the real model (skipped when it is not in the local cache) --------------------------------------
@pytest.fixture(scope="module")
def real():
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)
    except Exception:
        pytest.skip(f"{MODEL_NAME} is not in the local Hugging Face cache (run scripts.train_guard_classifier once)")
    classifier = LocalClinicalClassifier()
    classifier.warm()
    return classifier


def test_evaluation_cases_are_separated_with_the_rules(real):
    # The 26 cases are in no training set (the weights are trained on intake messages, seeds and the prompt table).
    golden, extra = load_cases()
    for case in golden + extra:
        must_refuse = case["category"] in ("guardrail", "injection")
        result = check_input(case["question"], real)
        assert result.blocked is must_refuse, (case["id"], result)


def test_the_two_cases_that_pass_the_rules_are_caught_by_the_classifier(real):
    _, extra = load_cases()
    for case in extra:
        if case["id"].endswith(("EX-003", "EX-006")):
            assert not check_rules(case["question"]).blocked
            result = check_input(case["question"], real)
            assert result.blocked and result.layer == "classifier" and result.score > 0.5


def test_allowed_prompts_are_not_blocked(real):
    # Not a hold-out: these prompts are part of the training negatives. It guards against a bad retrain.
    for message in ALLOWED_MESSAGES:
        assert check_input(message, real).blocked is False, message


def test_local_models_cap_the_cpu_threads_and_never_raise_them():
    torch = pytest.importorskip("torch")
    original = torch.get_num_threads()
    try:
        torch.set_num_threads(LOCAL_MODEL_THREADS + 8)  # a many-core default
        limit_local_model_threads()
        assert torch.get_num_threads() == LOCAL_MODEL_THREADS
        torch.set_num_threads(1)  # already lower: left alone
        limit_local_model_threads()
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(original)


def test_decisions_are_far_inside_the_time_limit(real):
    lines = (DATA_DIR / "intake" / "records.jsonl").read_text(encoding="utf-8").splitlines()[:40]
    slowest = 0.0
    for line in lines:
        started = time.perf_counter()
        result = check_input(json.loads(line)["raw_text"], real)
        slowest = max(slowest, time.perf_counter() - started)
        assert result.layer != "fail_closed"
    assert slowest < TIME_LIMIT_S / 4


def test_a_clinical_sentence_at_the_end_of_a_long_message_is_still_scored(tmp_path):
    classifier = LocalClinicalClassifier(_write_weights(tmp_path), _make_encoder({"dose": (1.0, 0.0), "slot": (0.2, 0.0)}))
    long_message = "Please find me a slot next week. " * 60 + "What dose should I take tonight?"
    assert classifier("Please find me a slot next week. " * 60).category == "none"
    assert classifier(long_message).category == "clinical"
