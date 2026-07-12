"""
Dense Embeddings Pipeline — RAG Wiki Pipeline
=============================================
Generates dense embeddings for all document chunks using fastembed
(all-MiniLM-L6-v2, 384-dim) and saves them locally.

Note: To push these embeddings to Qdrant Cloud, run upload_to_qdrant.py.

Setup
-----
    pip install fastembed pandas pyarrow

Usage
-----
    python embedding.py

Output
------
- data/dense_vectors.npy    : numpy array (n_chunks x 384), float32
- data/dense_meta.parquet   : doc_id / chunk_id / text, row-aligned with the array
"""

import os
import shutil
import numpy as np
import pandas as pd
from fastembed import TextEmbedding

DATA_DIR = 'data'
INPUT = os.path.join(DATA_DIR, 'clean', 'documents_chunked.parquet')
VECTORS_OUTPUT = os.path.join(DATA_DIR, 'embedding', 'dense_vectors.npy')
META_OUTPUT = os.path.join(DATA_DIR, 'embedding', 'dense_meta.parquet')

# all-MiniLM-L6-v2 via fastembed: 384-dim, fast, no PyTorch needed.
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
BATCH_SIZE = 64


def ensure_input_file():
    """Locate documents_chunked.parquet or raise a clear error."""
    os.makedirs(os.path.dirname(INPUT), exist_ok=True)
    os.makedirs(os.path.dirname(VECTORS_OUTPUT), exist_ok=True)
    if os.path.exists(INPUT):
        return

    candidates = [
        'documents_chunked.parquet',
        os.path.join(DATA_DIR, 'clean', 'documents_chunked.parquet'),
        os.path.join(DATA_DIR, 'documents_chunked (1).parquet'),
        'documents_chunked (1).parquet',
    ]
    found = next((c for c in candidates if os.path.exists(c)), None)

    if found:
        print(f"  [INFO] documents_chunked.parquet not found at {INPUT}; "
              f"found it at '{found}', copying it into place.")
        shutil.copy(found, INPUT)
    else:
        raise FileNotFoundError(
            f"Could not find 'documents_chunked.parquet' anywhere (looked in: "
            f"{[INPUT] + candidates}). Make sure you've run chunking.py first."
        )


def main():
    print("=" * 60)
    print("  RAG Wiki Pipeline — Dense Embedding (fastembed)")
    print("=" * 60)

    print("\nLocating input file...")
    ensure_input_file()

    print(f"\nLoading model '{MODEL_NAME}' (downloads on first run)...")
    model = TextEmbedding(model_name=MODEL_NAME)

    print("Loading chunked documents...")
    df = pd.read_parquet(INPUT)
    print(f"  Chunks: {len(df):,} rows")

    print(f"\nEncoding with batch_size={BATCH_SIZE} (this may take a minute)...")
    texts = df['document'].tolist()
    embeddings = list(model.embed(texts, batch_size=BATCH_SIZE))
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"  Matrix shape: {embeddings.shape}")

    np.save(VECTORS_OUTPUT, embeddings)
    print(f"\n  Saved: {VECTORS_OUTPUT}")

    df[['doc_id', 'chunk_id', 'document']].to_parquet(META_OUTPUT, index=False)
    print(f"  Saved: {META_OUTPUT}")

    print("\nDone. To upload to Qdrant Cloud, run: python upload_to_qdrant.py")


if __name__ == '__main__':
    main()
