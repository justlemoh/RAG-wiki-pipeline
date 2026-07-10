"""
Dense (neural) embeddings for RAG Wiki Pipeline - run this on YOUR machine
============================================================================
This is the production-quality alternative to embed_tfidf.py. It uses a real
sentence-embedding neural network instead of TF-IDF, which captures meaning
(synonyms, paraphrase) instead of just exact word overlap.

Requires internet access to download the model from Hugging Face the first
time (~90MB, cached locally after that) - this is why it couldn't be run in
the sandboxed environment used to build the rest of this pipeline.

Setup
-----
    pip install sentence-transformers pandas pyarrow numpy

Usage
-----
    python embed_dense.py

Output
------
- data/dense_vectors.npy    : numpy array (n_chunks x 384), float32
- data/dense_meta.parquet   : doc_id / chunk_id / text, row-aligned with the array
"""

import os
import shutil
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = 'data'
INPUT = os.path.join(DATA_DIR, 'documents_chunked.parquet')
VECTORS_OUTPUT = os.path.join(DATA_DIR, 'dense_vectors.npy')
META_OUTPUT = os.path.join(DATA_DIR, 'dense_meta.parquet')

# all-MiniLM-L6-v2: 384-dim, fast, a strong default for semantic search.
# For higher quality (slower, 768-dim), swap in 'all-mpnet-base-v2'.
MODEL_NAME = 'all-MiniLM-L6-v2'


def ensure_input_file():
    """
    Make sure documents_chunked.parquet exists inside DATA_DIR. This is the
    OUTPUT of the chunking step (Step 2) - if it's missing, chunking either
    wasn't run yet or saved somewhere unexpected. Look in a few likely spots
    before giving up, and give a clear error either way.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(INPUT):
        return

    candidates = [
        'documents_chunked.parquet',                              # ./documents_chunked.parquet
        os.path.join(DATA_DIR, 'documents_chunked (1).parquet'),  # duplicate-upload naming
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
            f"{[INPUT] + candidates}). Make sure you've run the chunking step "
            f"(Step 2) first - embedding runs on its output."
        )


def main():
    print("Locating input file...")
    ensure_input_file()

    print("Loading model (downloads on first run)...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading chunked documents...")
    df = pd.read_parquet(INPUT)
    print(f"  Chunks: {len(df):,} rows")

    print("Encoding (this may take a minute)...")
    embeddings = model.encode(
        df['document'].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # so cosine similarity == dot product
    )
    embeddings = embeddings.astype(np.float32)
    print(f"  Matrix shape: {embeddings.shape}")

    np.save(VECTORS_OUTPUT, embeddings)
    print(f"  Saved: {VECTORS_OUTPUT}")

    df[['doc_id', 'chunk_id', 'document']].to_parquet(META_OUTPUT, index=False)
    print(f"  Saved: {META_OUTPUT}")

    print("\nDone. To query this index, encode your query with the same")
    print("model.encode([query], normalize_embeddings=True) and take the")
    print("dot product against every row (cosine similarity, since both")
    print("sides are normalized) - same pattern as retrieve.py.")


if __name__ == '__main__':
    main()

