"""
Upload Embeddings to Qdrant Cloud
===================================
Run this script ONCE locally to embed all document chunks and upload them
to Qdrant Cloud. After this, the server uses Qdrant for retrieval — no
local .npy or .parquet files needed on the server.

Setup
-----
    pip install qdrant-client[fastembed] pandas pyarrow python-dotenv

Usage
-----
    python upload_to_qdrant.py

Requirements
------------
Make sure your .env file contains:
    QDRANT_URL=https://your-cluster.cloud.qdrant.io
    QDRANT_API_KEY=your-api-key
"""

import os
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Document

load_dotenv()

DATA_DIR = 'data'
INPUT = os.path.join(DATA_DIR, 'clean', 'documents_chunked.parquet')
COLLECTION_NAME = 'wiki_chunks'
# all-MiniLM-L6-v2 via fastembed (384-dim, fast, no PyTorch needed)
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
VECTOR_SIZE = 384
BATCH_SIZE = 64


def main():
    print("=" * 60)
    print("  RAG Wiki Pipeline — Upload to Qdrant Cloud")
    print("=" * 60)

    # Validate env vars
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        raise EnvironmentError(
            "QDRANT_URL and QDRANT_API_KEY must be set in your .env file."
        )

    # Load chunked documents
    if not os.path.exists(INPUT):
        raise FileNotFoundError(
            f"'{INPUT}' not found. Run chunking.py first."
        )
    print(f"\nLoading '{INPUT}'...")
    df = pd.read_parquet(INPUT)
    print(f"  Chunks: {len(df):,} rows")

    # Connect to Qdrant Cloud
    print(f"\nConnecting to Qdrant Cloud ({qdrant_url})...")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    print(f"  Embedding model : {MODEL_NAME}")

    # Delete and recreate the collection (clean re-upload)
    if client.collection_exists(COLLECTION_NAME):
        print(f"\n  [INFO] Collection '{COLLECTION_NAME}' exists — deleting and re-creating.")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"  Collection '{COLLECTION_NAME}' created.")

    # Build document and metadata lists
    documents = df['document'].tolist()
    metadata = [
        {'doc_id': int(row['doc_id']), 'chunk_id': int(row['chunk_id'])}
        for _, row in df.iterrows()
    ]

    # Upload in batches
    print(f"\nUploading {len(documents):,} chunks (batch_size={BATCH_SIZE})...")
    for start in range(0, len(documents), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(documents))
        batch_docs = documents[start:end]
        batch_meta = metadata[start:end]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=start + i,
                    vector=Document(text=doc, model=MODEL_NAME),
                    payload={**meta, 'document': doc},
                )
                for i, (doc, meta) in enumerate(zip(batch_docs, batch_meta))
            ],
        )
        print(f"  [{end:>5}/{len(documents)}] uploaded", end='\r')

    print(f"\n\n  Done! {len(documents):,} chunks stored in Qdrant Cloud.")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Cluster    : {qdrant_url}")
    print("\nYou can now use retrieve.py — it will query Qdrant directly.")


if __name__ == '__main__':
    main()
