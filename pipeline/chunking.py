"""
Document Chunking for RAG Wiki Pipeline
========================================
Chunks the cleaned documents for retrieval. Since the corpus is already made
of short, pre-segmented wiki passages (median ~54 words / ~70 tokens), most
documents are left untouched. Only long documents are split further, on
sentence boundaries, so no chunk cuts a sentence in half.

Strategy
--------
- Documents with <= LONG_DOC_THRESHOLD words are kept as a single chunk
  (this covers ~97% of the corpus).
- Documents above the threshold are split into chunks of ~CHUNK_SIZE_WORDS
  words, built from whole sentences, with an overlap of ~OVERLAP_WORDS words
  between consecutive chunks so context isn't lost at the boundary.
"""

import os
import re
import shutil
import pandas as pd

DATA_DIR = 'data'
INPUT = os.path.join(DATA_DIR, 'clean', 'documents_cleaned.parquet')
OUTPUT = os.path.join(DATA_DIR, 'clean', 'documents_chunked.parquet')
REPORT_OUTPUT = 'Chunking_Report.md'

LONG_DOC_THRESHOLD = 200   # words - docs at or below this stay as one chunk
CHUNK_SIZE_WORDS = 175     # target words per chunk when splitting
OVERLAP_WORDS = 30         # word overlap between consecutive chunks


def ensure_input_file():
    """
    Make sure documents_cleaned.parquet exists.
    """
    os.makedirs(os.path.dirname(INPUT), exist_ok=True)
    if os.path.exists(INPUT):
        return

    candidates = [
        'documents_cleaned.parquet',
        os.path.join(DATA_DIR, 'clean', 'documents_cleaned.parquet'),
        os.path.join(DATA_DIR, 'documents_cleaned (1).parquet'),
        'documents_cleaned (1).parquet',
    ]
    found = next((c for c in candidates if os.path.exists(c)), None)

    if found:
        print(f"  [INFO] documents_cleaned.parquet not found at {INPUT}; "
              f"found it at '{found}', copying it into place.")
        shutil.copy(found, INPUT)
    else:
        raise FileNotFoundError(
            f"Could not find 'documents_cleaned.parquet' anywhere (looked in: "
            f"{[INPUT] + candidates}). Make sure you've run the cleaning step "
            f"(Step 1) first - chunking runs on its output, not on the raw data."
        )


def split_sentences(text):
    """Split text into sentences, keeping the sentence-ending punctuation."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


def chunk_document(text, chunk_size=CHUNK_SIZE_WORDS, overlap=OVERLAP_WORDS):
    """
    Split a long document into overlapping chunks built from whole sentences.
    Falls back to a single chunk if the document has only one sentence longer
    than chunk_size (can't split without breaking mid-sentence).
    """
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [text]

    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent.split())
        if current and current_len + sent_len > chunk_size:
            chunks.append(' '.join(current))
            # build overlap from the tail of the current chunk
            overlap_sents = []
            overlap_len = 0
            for s in reversed(current):
                s_len = len(s.split())
                if overlap_len + s_len > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_len += s_len
            current = overlap_sents
            current_len = overlap_len
        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(' '.join(current))

    return chunks


def chunk_documents(df_doc):
    """Apply chunking to a documents dataframe. Returns new df + stats."""
    stats = {
        'original_count': len(df_doc),
        'docs_split': 0,
        'docs_unchanged': 0,
    }

    records = []
    for doc_id, row in df_doc.iterrows():
        text = row['document']
        word_count = len(text.split())

        if word_count <= LONG_DOC_THRESHOLD:
            stats['docs_unchanged'] += 1
            records.append({
                'doc_id': doc_id,
                'chunk_id': 0,
                'n_chunks': 1,
                'document': text,
            })
        else:
            stats['docs_split'] += 1
            chunks = chunk_document(text)
            for i, chunk in enumerate(chunks):
                records.append({
                    'doc_id': doc_id,
                    'chunk_id': i,
                    'n_chunks': len(chunks),
                    'document': chunk,
                })

    df_chunked = pd.DataFrame(records)
    stats['final_chunk_count'] = len(df_chunked)
    return df_chunked, stats


def generate_report(stats, df_chunked):
    word_counts = df_chunked['document'].str.split().apply(len)
    report = f"""# Chunking Report

## Strategy

- Documents with <= {LONG_DOC_THRESHOLD} words are kept as a single chunk
  (covers the large majority of the corpus, since these documents are
  already short pre-segmented wiki passages).
- Longer documents are split into ~{CHUNK_SIZE_WORDS}-word chunks built from
  whole sentences, with a ~{OVERLAP_WORDS}-word overlap between consecutive
  chunks.

## Summary

| Metric | Value |
|--------|-------|
| **Original documents** | {stats['original_count']:,} |
| **Documents kept as-is (<= {LONG_DOC_THRESHOLD} words)** | {stats['docs_unchanged']:,} |
| **Documents split into multiple chunks** | {stats['docs_split']:,} |
| **Total chunks produced** | {stats['final_chunk_count']:,} |

## Chunk size distribution (words)

| Percentile | Words |
|---|---|
| min | {word_counts.min()} |
| p25 | {word_counts.quantile(0.25):.0f} |
| p50 (median) | {word_counts.quantile(0.50):.0f} |
| p75 | {word_counts.quantile(0.75):.0f} |
| p95 | {word_counts.quantile(0.95):.0f} |
| max | {word_counts.max()} |

## Output

`data/documents_chunked.parquet` — columns: `doc_id` (original document
index), `chunk_id` (position within its parent document), `n_chunks` (total
chunks for that document), `document` (chunk text).
"""
    return report


def main():
    print("=" * 60)
    print("  RAG Wiki Pipeline - Document Chunking")
    print("=" * 60)

    print("\nLocating input file...")
    ensure_input_file()

    print("\nLoading cleaned documents...")
    df_doc = pd.read_parquet(INPUT)
    print(f"  Documents: {len(df_doc):,} rows")

    print("\n--- Chunking ---")
    df_chunked, stats = chunk_documents(df_doc)
    print(f"  Kept as single chunk: {stats['docs_unchanged']:,}")
    print(f"  Split into multiple chunks: {stats['docs_split']:,}")
    print(f"  Total chunks: {stats['final_chunk_count']:,}")

    print("\n--- Saving ---")
    df_chunked.to_parquet(OUTPUT, index=False)
    print(f"  Saved: {OUTPUT}")

    report = generate_report(stats, df_chunked)
    with open(REPORT_OUTPUT, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Saved: {REPORT_OUTPUT}")

    print("\nDone.")


if __name__ == '__main__':
    main()

