"""
Retrieval Module for RAG Wiki Pipeline
======================================
Loads pre-computed dense embeddings and performs semantic search using
cosine similarity.
"""

import os
import argparse
import numpy as np
import pandas as pd
from fastembed import TextEmbedding

DATA_DIR = 'data'
VECTORS_PATH = os.path.join(DATA_DIR, 'dense_vectors.npy')
META_PATH = os.path.join(DATA_DIR, 'dense_meta.parquet')
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# Global variables for caching model and data
_model = None
_vectors = None
_meta = None


def load_retrieval_resources():
    """Load the fastembed model, vector matrix, and metadata."""
    global _model, _vectors, _meta

    if _model is not None and _vectors is not None and _meta is not None:
        return _model, _vectors, _meta

    if not os.path.exists(VECTORS_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError(
            f"Embedding files not found. Make sure you ran embedding.py "
            f"to generate {VECTORS_PATH} and {META_PATH}."
        )

    # Load model (fastembed uses ONNX - no PyTorch required)
    _model = TextEmbedding(model_name=MODEL_NAME)

    # Load vectors (n_chunks x 384)
    _vectors = np.load(VECTORS_PATH)

    # Load metadata dataframe (doc_id, chunk_id, document)
    _meta = pd.read_parquet(META_PATH)

    return _model, _vectors, _meta


def retrieve(query, k=3):
    """
    Search the document chunks for the top-k most similar to the query.
    Returns a list of dicts containing: doc_id, chunk_id, document text, and similarity score.
    """
    model, vectors, meta = load_retrieval_resources()

    # Encode query using fastembed (returns a generator of numpy arrays)
    query_vector = list(model.embed([query]))[0]

    # Compute dot product against all rows
    scores = np.dot(vectors, query_vector)

    # Get the top-k indices
    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_indices:
        row = meta.iloc[idx]
        results.append({
            'doc_id': int(row['doc_id']),
            'chunk_id': int(row['chunk_id']),
            'document': row['document'],
            'score': float(scores[idx]),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Semantic Search Retrieval for RAG Wiki Pipeline")
    parser.add_argument("--query", type=str, help="The query string to search for")
    parser.add_argument("-k", type=int, default=3, help="Number of results to retrieve (default: 3)")
    args = parser.parse_args()

    # Load resources early
    print("Loading index resources...")
    try:
        load_retrieval_resources()
    except Exception as e:
        print(f"Error: {e}")
        return

    if args.query:
        print(f"\nSearching for: '{args.query}' (k={args.k})")
        results = retrieve(args.query, k=args.k)
        for i, res in enumerate(results, 1):
            print("-" * 60)
            print(f"Result {i} (Score: {res['score']:.4f}) | Doc {res['doc_id']} Chunk {res['chunk_id']}")
            print("-" * 60)
            print(res['document'])
        print("-" * 60)
    else:
        # Interactive loop
        print("\nEntering interactive mode. Type 'exit' or 'quit' to stop.")
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
