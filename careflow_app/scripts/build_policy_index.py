"""Build the M3 Qdrant policy index (out/qdrant) and report a retrieval baseline.

Run from careflow_app/:
  python -m scripts.build_policy_index            # real embeddings (CAREFLOW_EMBEDDING_MODEL, default Gemini)
  python -m scripts.build_policy_index --local    # offline hashing embedder, no API calls

LIVE cost with the real model: about 4 embedding requests for the corpus (50 chunks per request) and 1
for the evaluation queries. Vectors are cached in out/embeddings.sqlite, so a rebuild costs nothing.

Reported (the M4 starting point, not a gate):
- near-duplicate handbooks: for each of the 8 specialties, an unscoped question about that service line;
  the top hit must come from that specialty's handbook;
- golden set: for cases with must_cite, whether any / all cited documents are in the top 4 and top 5.
"""

from __future__ import annotations

import argparse
import json
import time

from careflow.config import DATA_DIR, QDRANT_PATH, get_settings
from careflow.rag.chunker import SPECIALTY_BY_SLUG, load_corpus_chunks
from careflow.rag.embeddings import HashingEmbedder, LiteLLMEmbedder
from careflow.rag.index import PolicyIndex

HANDBOOK_QUESTION = "How long is a new-patient {s} consultation, and what is its list price?"


def evaluate(index: PolicyIndex) -> None:
    titles = {s: f"{s} Service Line Handbook" for s in SPECIALTY_BY_SLUG.values()}
    golden = json.loads((DATA_DIR / "eval" / "golden_set.json").read_text(encoding="utf-8"))
    cited = [c for c in golden if c.get("must_cite")]
    questions = [HANDBOOK_QUESTION.format(s=s) for s in titles] + [c["question"] for c in cited]
    index.embedder.embed(questions)  # one batched request; searches below hit the cache

    right = 0
    print("\nNear-duplicate handbooks (unscoped search, top hit):")
    for s, title in titles.items():
        top = index.search(HANDBOOK_QUESTION.format(s=s), limit=1)
        ok = bool(top) and top[0].doc_title == title
        right += ok
        print(f"  [{'ok' if ok else 'MISS'}] {s:<17} -> {top[0].doc_title if top else '-'}")
    print(f"  {right}/{len(titles)} correct")

    # any_cite: at least one cited document retrieved; all_cite: every cited document retrieved (the
    # multi-hop cases need two). k=4 is what search_policies returns by default.
    score = {(m, k): 0 for m in ("any", "all") for k in (4, 5)}
    print("\nGolden set, must_cite retrieval (any / all cited documents in the top k):")
    for c in cited:
        found = {k: {h.doc_title for h in index.search(c["question"], limit=k)} for k in (4, 5)}
        marks = []
        for k in (4, 5):
            hit_any = any(t in found[k] for t in c["must_cite"])
            hit_all = all(t in found[k] for t in c["must_cite"])
            score[("any", k)] += hit_any
            score[("all", k)] += hit_all
            marks.append(f"@{k} {'all' if hit_all else 'any' if hit_any else 'MISS'}")
        print(f"  [{', '.join(marks)}] {c['id']} ({c['category']}): wants {c['must_cite']}")
    for k in (4, 5):
        print(f"  k={k}: any_cite {score[('any', k)]}/{len(cited)}   all_cite {score[('all', k)]}/{len(cited)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="offline hashing embedder (no API calls)")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    embedder = HashingEmbedder() if args.local else LiteLLMEmbedder(get_settings())
    chunks = load_corpus_chunks()
    print(f"embedder: {embedder.name}   chunks: {len(chunks)}   index: {QDRANT_PATH}")
    index = PolicyIndex.open(embedder)
    try:
        t = time.time()
        print(f"build: {index.build(chunks)} in {time.time() - t:.1f}s")
        if not args.no_eval:
            evaluate(index)
        if isinstance(embedder, LiteLLMEmbedder):
            print(f"\nembedding API requests this run: {embedder.calls}")
    finally:
        index.close()


if __name__ == "__main__":
    main()
