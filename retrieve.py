"""
Retrieval Module for RAG Wiki Pipeline
======================================
Queries Qdrant Cloud for the top-k most semantically similar document
chunks to the given query, using fastembed for query encoding.
"""

import os
import argparse
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

COLLECTION_NAME = 'wiki_chunks'
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# Cached client (initialized once per process)
_client = None


def get_qdrant_client():
    """Initialize and cache the Qdrant client with fastembed model."""
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
    _client.set_model(MODEL_NAME)
    return _client


def retrieve(query, k=3):
    """
    Search Qdrant Cloud for the top-k most similar chunks to the query.
    Returns a list of dicts: doc_id, chunk_id, document text, similarity score.
    """
    client = get_qdrant_client()

    results = client.query(
        collection_name=COLLECTION_NAME,
        query_text=query,
        limit=k,
    )

    return [
        {
            'doc_id': r.metadata.get('doc_id'),
            'chunk_id': r.metadata.get('chunk_id'),
            'document': r.document,
            'score': float(r.score),
        }
        for r in results
    ]


def main():
    parser = argparse.ArgumentParser(description="Semantic Search — Qdrant Cloud")
    parser.add_argument("--query", type=str, help="The query string to search for")
    parser.add_argument("-k", type=int, default=3, help="Number of results to retrieve (default: 3)")
    args = parser.parse_args()

    print("Connecting to Qdrant Cloud...")
    try:
        get_qdrant_client()
        print("  Connected.\n")
    except Exception as e:
        print(f"Error: {e}")
        return

    if args.query:
        print(f"Searching for: '{args.query}' (k={args.k})")
        results = retrieve(args.query, k=args.k)
        for i, res in enumerate(results, 1):
            print("-" * 60)
            print(f"Result {i} (Score: {res['score']:.4f}) | Doc {res['doc_id']} Chunk {res['chunk_id']}")
            print("-" * 60)
            print(res['document'])
        print("-" * 60)
    else:
        print("Entering interactive mode. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                query = input("\nQuery > ").strip()
                if not query:
                    continue
                if query.lower() in ['exit', 'quit']:
                    break
                results = retrieve(query, k=args.k)
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
