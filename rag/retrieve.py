"""
Retrieval Module for RAG Wiki Pipeline
======================================
Queries Qdrant Cloud for the top-k most semantically similar document
chunks to the given query, using fastembed for query encoding.

Optionally applies cross-encoder reranking (enabled by default) to
improve retrieval quality. The reranker first retrieves a larger
candidate pool and then re-scores each candidate using a lightweight
cross-encoder model (Xenova/ms-marco-MiniLM-L-6-v2, ~80MB, CPU-only).
"""

import os
import argparse
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Document
from fastembed.rerank.cross_encoder import TextCrossEncoder

load_dotenv()

COLLECTION_NAME = 'wiki_chunks'
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
RERANKER_MODEL_NAME = 'Xenova/ms-marco-MiniLM-L-6-v2'

# Cached client and reranker (initialized once per process)
_client = None
_reranker = None


def get_qdrant_client():
    """Initialize and cache the Qdrant client."""
    global _client
    if _client is not None:
        return _client

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        raise EnvironmentError(
            "QDRANT_URL and QDRANT_API_KEY must be set in your environment or .env file."
        )

    _client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    return _client


def get_reranker():
    """Initialize and cache the cross-encoder reranker model."""
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(model_name=RERANKER_MODEL_NAME)
    return _reranker


def retrieve(query, k=3, rerank=True):
    """
    Search Qdrant Cloud for the top-k most similar chunks to the query.
    Returns a list of dicts: doc_id, chunk_id, document text, score.

    When rerank=True (default), a larger candidate pool is first fetched
    (max(k * 3, 10) candidates) and then re-scored using a lightweight
    cross-encoder model. The final score reflects the cross-encoder score.
    When rerank=False, falls back to pure vector similarity (original behaviour).
    """
    client = get_qdrant_client()

    # Determine how many candidates to fetch from Qdrant
    fetch_limit = max(k * 3, 10) if rerank else k

    qdrant_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=Document(text=query, model=MODEL_NAME),
        limit=fetch_limit,
    )

    candidates = [
        {
            'doc_id': p.payload.get('doc_id'),
            'chunk_id': p.payload.get('chunk_id'),
            'document': p.payload.get('document'),
            'score': float(p.score),
        }
        for p in qdrant_results.points
    ]

    if not rerank or not candidates:
        return candidates

    # --- Reranking stage ---
    # reranker.rerank() returns a list of float scores, one per document,
    # in the same order as the input documents list.
    reranker = get_reranker()
    documents = [c['document'] for c in candidates]
    rerank_scores = list(reranker.rerank(query, documents))

    # Attach cross-encoder score to each candidate and sort descending
    for candidate, score in zip(candidates, rerank_scores):
        candidate['score'] = float(score)

    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:k]


def main():
    parser = argparse.ArgumentParser(description="Semantic Search — Qdrant Cloud (with optional reranking)")
    parser.add_argument("--query", type=str, help="The query string to search for")
    parser.add_argument("-k", type=int, default=3, help="Number of results to retrieve (default: 3)")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking (uses raw vector similarity)")
    args = parser.parse_args()

    use_rerank = not args.no_rerank
    mode_label = "vector-only (no rerank)" if not use_rerank else f"reranked with '{RERANKER_MODEL_NAME}'"

    print("Connecting to Qdrant Cloud...")
    try:
        get_qdrant_client()
        print("  Connected.\n")
    except Exception as e:
        print(f"Error: {e}")
        return

    if args.query:
        print(f"Searching for: '{args.query}' (k={args.k}, mode={mode_label})")
        results = retrieve(args.query, k=args.k, rerank=use_rerank)
        for i, res in enumerate(results, 1):
            print("-" * 60)
            print(f"Result {i} (Score: {res['score']:.4f}) | Doc {res['doc_id']} Chunk {res['chunk_id']}")
            print("-" * 60)
            print(res['document'])
        print("-" * 60)
    else:
        print(f"Entering interactive mode [{mode_label}]. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                query = input("\nQuery > ").strip()
                if not query:
                    continue
                if query.lower() in ['exit', 'quit']:
                    break
                results = retrieve(query, k=args.k, rerank=use_rerank)
                for i, res in enumerate(results, 1):
                    print("-" * 60)
                    print(f"Result {i} (Score: {res['score']:.4f}) | Doc {res['doc_id']} Chunk {res['chunk_id']}")
                    print("-" * 60)
                    print(res['document'])
                print("-" * 60)
            except KeyboardInterrupt:
                print()
                break


if __name__ == '__main__':
    main()
