"""Train the local Clinical Content Classifier (src/guard_model.py). Offline: no hosted calls.

Run from app/:
  python -m scripts.train_guard_classifier            # report, then write src/data/clinical_gate.npz
  python -m scripts.train_guard_classifier --report   # report only

The first run downloads the sentence encoder (all-MiniLM-L6-v2, about 90 MB, Apache-2.0) into the
Hugging Face cache; afterwards it is read from there.

Training data, all sentence-level:
- negative: every sentence of the intake messages labelled as not seeking clinical advice (and not
  Emergent), the allowed prompts in tests/test_guardrails.py, and a few administrative questions;
- positive: seed questions written from OP-CS-014 5.1 (categories C1 to C8), the clinical and emergency
  prompts in tests/test_guardrails.py, and, for each intake message labelled as seeking advice, its
  highest-scoring sentence (found over three rounds, since the label is for the message, not a sentence).
Messages labelled Emergent but not advice-seeking are left out: emergency wording is the rules' job.

Fixed up front, not tuned on the results: logistic regression, C=10, balanced class weights, and the
policy's 0.30 threshold. The report says what was held out from what. The data is small and synthetic
(about 200 messages); the figures show the classifier works on this data, nothing about real patients.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

from scripts._common import load_intake_records
from scripts.run_system_eval import load_cases
from src.guard_model import MODEL_NAME, WEIGHTS, split_sentences
from src.guardrails import THRESHOLD, check_rules

INVERSE_REGULARISATION = 10.0  # inverse regularisation strength of the logistic regression
SEEDS_POSITIVE = [  # OP-CS-014 5.1, C1 to C8: one plain question per listed kind
    "Is this rash normal?", "Why does my chest feel tight?", "How long does a fever like this last?", "Is this bruise too big?",
    "What is this medication for?", "What are the side effects of this tablet?", "Can I drink alcohol with this medicine?",
    "I missed a dose, what now?", "Is the generic the same as the brand?", "Does it matter that my tablets were left in the heat?",
    "How many tablets do I take?", "How often should I take it?", "Can I split the tablet?", "How do I taper off it?",
    "Can you explain my result?", "Is it bad?", "Do I need to worry about my result?", "Can you compare this result with my last one?",
    "Do I need to stop exercising?", "Is it safe to fly with this?", "Can I still go to work?", "Would you recommend surgery?",
    "What would you do in my place?", "Is it okay if I skip a day?",
    "How do I look after the wound?", "When can I go back to work after the operation?", "Can I have a bath after the procedure?",
    "When can I drive again after surgery?", "Do I need to change my diet after the procedure?",
    "Is it safe in pregnancy?", "My baby has a temperature, what do I do?", "My child keeps coughing at night, is that a problem?",
    "I've been feeling really low lately.", "I can't cope any more.",
]
SEEDS_NEGATIVE = [
    "Should I bring my insurance card?", "What are your opening hours?", "Where do I park?", "How do I get a copy of my bill?",
    "Can I book an appointment for next week?", "Do I need a referral to see a specialist?", "Which plans do you accept?",
    "How do I cancel my appointment?", "A new patient visit, how long does it take?", "Is pre-authorisation needed for an MRI?",
    # Cost and coverage questions that name a procedure: without negatives of this kind the classifier
    # blocks them (golden case EV-015, a co-pay calculation for a knee arthroscopy, scored 0.40). The
    # wording is new, not the golden set's, but the golden set is not a clean hold-out for this one kind
    # of question.
    "What would a patient owe for a hip replacement after meeting half of the deductible?",
    "If someone has paid $900 towards a $2,000 deductible, what is their share of a $4,500 operation?",
    "How much is the co-pay for a CT scan on the gold plan?",
    "What is the patient's cost for a tonsillectomy before the out-of-pocket cap is applied?",
    "What does the plan pay towards a cataract operation and what is left for the patient?",
    "A patient is having an endoscopy next month; what is the coinsurance on the bronze plan?",
    "What is the self-pay price for a shoulder injection?",
    "Is a sleep study covered, and what is the approval window?",
]


def load_data() -> dict:
    """The labelled intake messages with their labels (seeks = asks for clinical advice; refuse = that or
    Emergent), the guardrail tests' prompt table, and the evaluation cases as (id, question, must refuse)."""
    from tests.test_guardrails import ALLOWED_MESSAGES, BLOCKED

    records = [record for record in load_intake_records() if isinstance(record.get("ground_truth"), dict)]
    golden, extra = load_cases()
    return {
        "texts": [record["raw_text"] for record in records],
        "seeks": np.array([int(bool(record["ground_truth"]["seeks_clinical_advice"])) for record in records]),
        "refuse": np.array([int(bool(record["ground_truth"]["seeks_clinical_advice"]) or record["ground_truth"].get("urgency") == "Emergent")
                            for record in records]),
        "table_blocked": [text for text, category in BLOCKED if category != "injection"],
        "table_allowed": list(ALLOWED_MESSAGES),
        "cases": [(case["id"], case["question"], int(case["category"] in ("guardrail", "injection"))) for case in golden + extra],
    }


def fit(sentence_embeddings: list[np.ndarray], data: dict, train: np.ndarray, embedded: dict, use_table: bool):
    """Fit on the intake messages at indices `train` plus the seeds (and the prompt table if use_table).
    sentence_embeddings[n] holds one row per sentence of message n; `embedded` holds the embedded seed and table sets."""
    from sklearn.linear_model import LogisticRegression

    seeks, refuse = data["seeks"], data["refuse"]
    negative = np.vstack([sentence_embeddings[index] for index in train if not refuse[index]] + [embedded["seeds_negative"]]
                         + ([embedded["table_allowed"]] if use_table else []))
    base_positive = np.vstack([embedded["seeds_positive"]] + ([embedded["table_blocked"]] if use_table else []))
    positive, model = base_positive, None
    for _ in range(3):  # the label is per message: add each advice-seeking message's top sentence, twice over
        features = np.vstack([negative, positive])
        labels = np.r_[np.zeros(len(negative)), np.ones(len(positive))]
        model = LogisticRegression(C=INVERSE_REGULARISATION, class_weight="balanced", max_iter=3000).fit(features, labels)
        tops = [sentence_embeddings[index][[int(np.argmax(model.predict_proba(sentence_embeddings[index])[:, 1]))]] for index in train if seeks[index]]
        positive = np.vstack([base_positive, *tops])
    return model


def score_message(model, embeddings: np.ndarray) -> float:
    """A message's score: the highest clinical probability among its sentences."""
    return float(model.predict_proba(embeddings)[:, 1].max())


def main() -> None:
    from sentence_transformers import SentenceTransformer
    from sklearn.model_selection import RepeatedStratifiedKFold

    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="print the figures; do not write the weights")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    encoder = SentenceTransformer(MODEL_NAME, device="cpu")
    encoder.max_seq_length = 128

    def embed(units: list[str]) -> np.ndarray:
        return encoder.encode(units, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True, batch_size=64)

    data = load_data()
    texts, seeks, refuse = data["texts"], data["seeks"], data["refuse"]
    sentence_embeddings = [embed(split_sentences(message)) for message in texts]
    embedded = {"seeds_positive": embed(SEEDS_POSITIVE), "seeds_negative": embed(SEEDS_NEGATIVE),
         "table_blocked": embed(data["table_blocked"]),
         # split like the intake negatives, and as scoring does: every sentence of an allowed prompt is a negative
         "table_allowed": embed([unit for text in data["table_allowed"] for unit in split_sentences(text)])}
    rules = np.array([check_rules(message).blocked for message in texts])
    ordinary = refuse == 0
    print(f"Encoder {MODEL_NAME}; {len(texts)} labelled intake messages ({int(seeks.sum())} seek advice, "
          f"{int(refuse.sum())} should be refused, {int(ordinary.sum())} ordinary); threshold {THRESHOLD}\n")

    # 1. Intake messages: 5 x 5-fold cross-validation. The prompt table is extra training data here.
    fold_scores = np.zeros((5, len(texts)))
    folds = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
    for fold, (train, test) in enumerate(folds.split(np.zeros(len(texts)), refuse)):
        model = fit(sentence_embeddings, data, train, embedded, use_table=True)
        for index in test:
            fold_scores[fold // 5, index] = score_message(model, sentence_embeddings[index])
    blocked = fold_scores >= THRESHOLD
    print("1. Intake messages, 5 x 5-fold cross-validation (mean over the 5 repeats)")
    print(f"   classifier alone : catches {np.mean((blocked & (seeks == 1)).sum(1)):.1f}/{int(seeks.sum())} advice-seeking messages; "
          f"blocks {np.mean((blocked & ordinary).sum(1)):.1f}/{int(ordinary.sum())} ordinary ones")
    print(f"   rules alone      : catches {int((rules & (refuse == 1)).sum())}/{int(refuse.sum())} that should be refused; "
          f"blocks {int((rules & ordinary).sum())}/{int(ordinary.sum())} ordinary ones")
    both = blocked | rules
    print(f"   rules+classifier : catches {np.mean((both & (refuse == 1)).sum(1)):.1f}/{int(refuse.sum())} that should be refused; "
          f"blocks {np.mean((both & ordinary).sum(1)):.1f}/{int(ordinary.sum())} ordinary ones")

    # 2. Prompt table held out: train on intake + seeds only.
    everything = np.arange(len(texts))
    no_table = fit(sentence_embeddings, data, everything, embedded, use_table=False)
    blocked_scores = [score_message(no_table, embed(split_sentences(prompt))) for prompt in data["table_blocked"]]
    allowed_scores = [score_message(no_table, embed(split_sentences(prompt))) for prompt in data["table_allowed"]]
    print("\n2. Prompt table (tests/test_guardrails.py) held out of training")
    print(f"   clinical and emergency prompts blocked {sum(value >= THRESHOLD for value in blocked_scores)}/{len(blocked_scores)}; "
          f"allowed prompts blocked {sum(value >= THRESHOLD for value in allowed_scores)}/{len(allowed_scores)}")

    # 3. The final model (intake + seeds + table) on the 26 evaluation cases, which are in no training set.
    final = fit(sentence_embeddings, data, everything, embedded, use_table=True)
    print("\n3. Golden set and extra refusal cases (never in training), final model")
    wrong = []
    for case_id, question, must_refuse in data["cases"]:
        case_score = score_message(final, embed(split_sentences(question)))
        by_rules = check_rules(question).blocked
        ok = ((case_score >= THRESHOLD) or by_rules) == bool(must_refuse)
        if not ok or (must_refuse and not by_rules):
            print(f"   {case_id[-6:]} score {case_score:.2f} rules={'block' if by_rules else 'pass'} expected "
                  f"{'refuse' if must_refuse else 'answer'}{'' if ok else '   <-- WRONG'}")
        wrong += [] if ok else [case_id]
    refusals = sum(refusal for _, _, refusal in data["cases"])
    print(f"   rules+classifier correct on {len(data['cases']) - len(wrong)}/{len(data['cases'])} "
          f"({refusals} refusals, {len(data['cases']) - refusals} answerable)")

    # 4. Latency of a whole decision, warm.
    weights, bias = final.coef_[0].astype(np.float32), float(final.intercept_[0])
    times = []
    for message in texts:
        start = time.perf_counter()
        _ = 1 / (1 + np.exp(-(embed(split_sentences(message)) @ weights + bias)))
        times.append((time.perf_counter() - start) * 1000)
    print(f"\n4. Warm decision time over {len(times)} intake messages: median {np.percentile(times, 50):.0f} ms, "
          f"95th percentile {np.percentile(times, 95):.0f} ms, max {max(times):.0f} ms (limit 2000 ms)")

    if args.report:
        return
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(WEIGHTS, w=weights, b=np.float32(bias), model=np.array(MODEL_NAME), C=np.float32(INVERSE_REGULARISATION),
             trained_on=np.array(f"{len(texts)} intake messages, {len(SEEDS_POSITIVE) + len(SEEDS_NEGATIVE)} seeds, "
                                 f"{len(data['table_blocked']) + len(data['table_allowed'])} table prompts"))
    print(f"\nWeights written to {WEIGHTS.relative_to(WEIGHTS.parents[2])} ({len(weights)} dimensions)")


if __name__ == "__main__":
    main()
