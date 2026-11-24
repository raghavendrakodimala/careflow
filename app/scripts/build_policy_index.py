"""Build the Qdrant policy index (out/qdrant) that search_policies serves from.

Run from app/:
  python -m scripts.build_policy_index                    # PDF copy (CAREFLOW_POLICY_SOURCE), real embeddings
  python -m scripts.build_policy_index --source markdown  # the clean markdown copy instead
  python -m scripts.build_policy_index --local            # offline hashing embedder, no API calls

LIVE cost with the real embedding model: about 4 requests (50 chunks each) the first time a copy is
built; vectors are cached in out/embeddings.sqlite, so a rebuild costs nothing. Each point gets a dense
vector and a BM25 sparse vector. Retrieval quality is measured by scripts/run_retrieval_eval.py.
"""

from __future__ import annotations

import argparse
import time

from src.config import QDRANT_PATH, get_settings
from src.rag.chunker import SOURCES, load_corpus_chunks
from src.rag.embeddings import HashingEmbedder, LiteLLMEmbedder
from src.rag.index import PolicyIndex


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=SOURCES, default=settings.careflow_policy_source)
    parser.add_argument("--local", action="store_true", help="offline hashing embedder (no API calls)")
    args = parser.parse_args()

    embedder = HashingEmbedder() if args.local else LiteLLMEmbedder(settings)
    chunks = load_corpus_chunks(source=args.source)
    print(f"source: {args.source}   embedder: {embedder.name}   chunks: {len(chunks)}   index: {QDRANT_PATH}")
    index = PolicyIndex.open(embedder)
    try:
        started = time.time()
        print(f"build: {index.build(chunks)} in {time.time() - started:.1f}s")
        # Smoke test: one fixed question, to show the fresh index answers.
        top = index.search("How quickly must a Priority referral be actioned?", limit=1)
        print(f"smoke: {top[0].doc_title} / {top[0].heading}" if top else "smoke: no result")
        if isinstance(embedder, LiteLLMEmbedder):
            print(f"embedding API requests this run: {embedder.calls}")
    finally:
        index.close()


if __name__ == "__main__":
    main()
