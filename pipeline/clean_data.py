"""
Data Cleaning Pipeline for RAG Wiki Pipeline
=============================================
"""

import os
import re
import shutil
import pandas as pd

DATA_DIR = 'data'
DOC_INPUT = os.path.join(DATA_DIR, 'raw', 'documents.parquet')
Q_INPUT = os.path.join(DATA_DIR, 'raw', 'questions.parquet')
DOC_OUTPUT = os.path.join(DATA_DIR, 'clean', 'documents_cleaned.parquet')
Q_OUTPUT = os.path.join(DATA_DIR, 'clean', 'questions_cleaned.parquet')
REPORT_OUTPUT = 'Cleaning_Report.md'

MIN_WORD_COUNT = 5


def ensure_input_files():
    """
    Make sure documents.parquet / questions.parquet exist inside DATA_DIR.
    If they're missing there but present elsewhere (e.g. dropped straight
    into the project root, or saved with a "(1)" suffix from a duplicate
    download), find them and copy/rename them into place instead of failing.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    for target_name, target_path in [
        ('documents.parquet', DOC_INPUT),
        ('questions.parquet', Q_INPUT),
    ]:
        if os.path.exists(target_path):
            continue  # already where it should be

        candidates = [
            target_name,                                   # ./documents.parquet
            os.path.join(DATA_DIR, 'raw', target_name),
            os.path.join(DATA_DIR, target_name),
            os.path.join(DATA_DIR, 'raw', target_name.replace('.parquet', ' (1).parquet')),
            target_name.replace('.parquet', ' (1).parquet'),
        ]
        found = next((c for c in candidates if os.path.exists(c)), None)

        if found:
            print(f"  [INFO] {target_name} not found at {target_path}; "
                  f"found it at '{found}', copying it into place.")
            shutil.copy(found, target_path)
        else:
            raise FileNotFoundError(
                f"Could not find '{target_name}' anywhere. Looked in: {candidates}. "
                f"Place documents.parquet and questions.parquet in the 'data/raw/' "
                f"folder next to this script before running it."
            )


def fix_mojibake(text):
    try:
        return text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def balance_parens(text):
    """Remove unmatched '(' or ')' left over after noise stripping."""
    stack = []
    remove = set()
    for i, ch in enumerate(text):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                stack.pop()
            else:
                remove.add(i)
    remove.update(stack)
    return ''.join(c for i, c in enumerate(text) if i not in remove)


def clean_text_noise(text):
    text = re.sub(
        r'\(\s*(?:official\s+full\s+name\s+in\s+)?;\s*pron\.\s*,\s*',
        '(', text
    )
    # Broader case: any semicolon right after an opening paren
    # e.g. "Theodore Roosevelt, Jr. ( ; October 27 1858 January 6 1919)"
    #   -> "Theodore Roosevelt, Jr. (October 27 1858 January 6 1919)"
    text = re.sub(r'\(\s*;\s*', '(', text)
    # Doubled/nested opening parens with stray commas
    # e.g. "Liechtenstein (  ( , ( ) is a tiny..." -> "Liechtenstein ("
    text = re.sub(r'\(\s*\(\s*,?\s*', '(', text)
    text = re.sub(r'\(\s*,\s*', '(', text)
    noise_pattern = re.compile(
        r'\(\s*(?:official\s+full\s+name\s+in\s+|pron\.|singular|'
        r'[a-zA-Z\s]+:\s*|[;,\s])*?\s*\)'
        r'|'
        r'\[\s*(?:official\s+full\s+name\s+in\s+|pron\.|singular|'
        r'[a-zA-Z\s]+:\s*|[;,\s])*?\s*\]'
    )
    prev_text = None
    while prev_text != text:
        prev_text = text
        text = noise_pattern.sub('', text)
    text = balance_parens(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,;?!])', r'\1', text)
    text = re.sub(r'^[,;.\s]+', '', text)
    text = text.strip()
    return text


def clean_documents(df_doc):
    stats = {
        'original_count': len(df_doc),
        'mojibake_fixed': 0,
        'noise_cleaned': 0,
        'unbalanced_parens_fixed': 0,
        'duplicates_removed': 0,
        'short_docs_removed': 0,
    }
    print("  [1/4] Fixing mojibake (double-encoded UTF-8)...")
    df_doc = df_doc.copy()
    original_texts = df_doc['document'].copy()
    df_doc['document'] = df_doc['document'].apply(fix_mojibake)
    stats['mojibake_fixed'] = int((df_doc['document'] != original_texts).sum())

    print("  [2/4] Cleaning text noise and empty parentheses...")
    original_texts = df_doc['document'].copy()
    unbalanced_before = original_texts.apply(lambda t: t.count('(') != t.count(')'))
    df_doc['document'] = df_doc['document'].apply(clean_text_noise)
    stats['noise_cleaned'] = int((df_doc['document'] != original_texts).sum())
    unbalanced_after = df_doc['document'].apply(lambda t: t.count('(') != t.count(')'))
    stats['unbalanced_parens_fixed'] = int((unbalanced_before & ~unbalanced_after).sum())

    print("  [3/4] Removing duplicate documents...")
    count_before = len(df_doc)
    df_doc = df_doc.drop_duplicates(subset='document', keep='first')
    stats['duplicates_removed'] = count_before - len(df_doc)

    print("  [4/4] Filtering short documents (<={} words)...".format(MIN_WORD_COUNT))
    df_doc['_word_count'] = df_doc['document'].str.split().apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    short_mask = df_doc['_word_count'] <= MIN_WORD_COUNT
    stats['short_docs_removed'] = int(short_mask.sum())
    df_doc = df_doc[~short_mask].drop(columns=['_word_count'])

    df_doc = df_doc.reset_index(drop=True)
    stats['final_count'] = len(df_doc)
    return df_doc, stats


GARBAGE_QUESTION_INDICES = [254, 802]
UNANSWERABLE_INDICES = [899]


def fix_s08_corruption(text):
    return text.replace('S08_', '')


def normalize_yes_no(gt):
    cleaned = gt.strip().rstrip('.!?').strip().lower()
    if cleaned == 'yes':
        return 'yes'
    elif cleaned == 'no':
        return 'no'
    return gt


def clean_ground_truth(gt):
    gt = gt.strip()
    if gt == 'slavory issues':
        gt = 'slavery issues'
    if gt.startswith('176.215 km'):
        gt = '176,215 km²'
    return gt


def format_question(q):
    q = q.strip()
    q = re.sub(r'\s+', ' ', q)  # collapse duplicate/leftover whitespace
    q = re.sub(r'\s*\(:\s*;\s*', ' ', q)
    q = q.replace('herbivour', 'herbivore')
    if not q.endswith('?'):
        if re.search(r'\?\s*["\']?\s*$', q):
            pass
        else:
            q = q.rstrip('.')
            q = q + '?'
    q = fix_s08_corruption(q)
    return q


def clean_questions(df_q):
    stats = {
        'original_count': len(df_q),
        's08_fixed_questions': 0,
        's08_fixed_ground_truths': 0,
        'yes_no_normalized': 0,
        'questions_reformatted': 0,
        'garbage_removed': 0,
        'unanswerable_flagged': 0,
    }
    df_q = df_q.copy()

    print("  [1/6] Removing garbage entries...")
    garbage_mask = df_q.index.isin(GARBAGE_QUESTION_INDICES)
    stats['garbage_removed'] = int(garbage_mask.sum())
    df_q = df_q[~garbage_mask].copy()

    print("  [2/6] Fixing S08_ corruption...")
    q_orig = df_q['question'].copy()
    gt_orig = df_q['ground_truth'].copy()
    df_q['question'] = df_q['question'].apply(fix_s08_corruption)
    df_q['ground_truth'] = df_q['ground_truth'].apply(fix_s08_corruption)
    stats['s08_fixed_questions'] = int((df_q['question'] != q_orig).sum())
    stats['s08_fixed_ground_truths'] = int((df_q['ground_truth'] != gt_orig).sum())

    print("  [3/6] Cleaning ground truth answers...")
    df_q['ground_truth'] = df_q['ground_truth'].apply(clean_ground_truth)

    print("  [4/6] Normalizing yes/no answers...")
    gt_before = df_q['ground_truth'].copy()
    df_q['ground_truth'] = df_q['ground_truth'].apply(normalize_yes_no)
    stats['yes_no_normalized'] = int((df_q['ground_truth'] != gt_before).sum())

    print("  [5/6] Formatting questions...")
    q_before = df_q['question'].copy()
    df_q['question'] = df_q['question'].apply(format_question)
    stats['questions_reformatted'] = int((df_q['question'] != q_before).sum())

    print("  [6/6] Flagging unanswerable entries...")
    for idx in UNANSWERABLE_INDICES:
        if idx in df_q.index:
            df_q.loc[idx, 'ground_truth'] = 'unanswerable'
            stats['unanswerable_flagged'] += 1

    df_q = df_q.reset_index(drop=True)
    stats['final_count'] = len(df_q)
    return df_q, stats


STOP_WORDS = set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an',
    'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
    'being', 'below', 'between', 'both', 'but', 'by', 'can', 'cannot',
    'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few',
    'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her',
    'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if',
    'in', 'into', 'is', 'it', 'its', 'itself', 'me', 'more', 'most', 'my',
    'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or',
    'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
    'same', 'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the',
    'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these',
    'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
    'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while',
    'who', 'whom', 'why', 'with', 'would', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'will', 'shall', 'may', 'might', 'must',
])


def check_alignment(df_doc, df_q):
    print("  Checking alignment (this may take a moment)...")
    doc_corpus = [doc.lower() for doc in df_doc['document']]
    unanswerable = []
    for idx, row in df_q.iterrows():
        q = row['question']
        gt = row['ground_truth']
        if gt == 'unanswerable':
            continue
        words = re.findall(r'\b\w+\b', q.lower())
        keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        if not keywords:
            continue
        min_match = max(1, len(keywords) // 2)
        found = False
        for doc in doc_corpus:
            match_count = sum(1 for kw in keywords if kw in doc)
            if match_count >= min_match:
                found = True
                break
        if not found:
            unanswerable.append((idx, q, gt))
    return unanswerable


def generate_report(doc_stats, q_stats, alignment_issues):
    report = f"""# Data Cleaning Report

This report summarizes the results of the data cleaning pipeline for the RAG Wiki Pipeline.

---

## 1. Documents Cleaning Summary

| Metric | Value |
|--------|-------|
| **Original document count** | {doc_stats['original_count']:,} |
| **Mojibake (encoding) fixes applied** | {doc_stats['mojibake_fixed']:,} |
| **Documents with noise cleaned** | {doc_stats['noise_cleaned']:,} |
| **Unbalanced parentheses fixed** | {doc_stats['unbalanced_parens_fixed']:,} |
| **Duplicate documents removed** | {doc_stats['duplicates_removed']:,} |
| **Short documents filtered (≤{MIN_WORD_COUNT} words)** | {doc_stats['short_docs_removed']:,} |
| **Final document count** | {doc_stats['final_count']:,} |

---

## 2. Questions Cleaning Summary

| Metric | Value |
|--------|-------|
| **Original question count** | {q_stats['original_count']:,} |
| **Garbage entries removed** | {q_stats['garbage_removed']:,} |
| **S08_ corruptions fixed (questions)** | {q_stats['s08_fixed_questions']:,} |
| **S08_ corruptions fixed (ground truths)** | {q_stats['s08_fixed_ground_truths']:,} |
| **Yes/No answers normalized** | {q_stats['yes_no_normalized']:,} |
| **Questions reformatted** | {q_stats['questions_reformatted']:,} |
| **Entries flagged as unanswerable** | {q_stats['unanswerable_flagged']:,} |
| **Final question count** | {q_stats['final_count']:,} |

---

## 3. QA-Document Alignment Check

"""
    if alignment_issues:
        report += f"**{len(alignment_issues)} potential alignment issues found:**\n\n"
        for idx, q, gt in alignment_issues:
            report += f"- **Q{idx}**: `{q}` → `{gt}`\n"
    else:
        report += "**All questions have matching document support. ✅**\n"

    report += f"""
---

## 4. Output Files

| File | Description |
|------|-------------|
| `data/documents_cleaned.parquet` | Cleaned documents ({doc_stats['final_count']:,} rows) |
| `data/questions_cleaned.parquet` | Cleaned questions ({q_stats['final_count']:,} rows) |
"""
    return report


def main():
    print("=" * 60)
    print("  RAG Wiki Pipeline - Data Cleaning")
    print("=" * 60)

    print("\nLocating input files...")
    ensure_input_files()

    print("\nLoading datasets...")
    df_doc = pd.read_parquet(DOC_INPUT)
    df_q = pd.read_parquet(Q_INPUT)
    print(f"  Documents: {len(df_doc):,} rows")
    print(f"  Questions: {len(df_q):,} rows")

    print("\n--- Cleaning Documents ---")
    df_doc_clean, doc_stats = clean_documents(df_doc)
    print(f"  [OK] Documents cleaned: {doc_stats['original_count']:,} -> {doc_stats['final_count']:,}")

    print("\n--- Cleaning Questions ---")
    df_q_clean, q_stats = clean_questions(df_q)
    print(f"  [OK] Questions cleaned: {q_stats['original_count']:,} -> {q_stats['final_count']:,}")

    print("\n--- Checking QA-Document Alignment ---")
    alignment_issues = check_alignment(df_doc_clean, df_q_clean)
    if alignment_issues:
        print(f"  [WARN] {len(alignment_issues)} potential alignment issues found")
        for idx, q, gt in alignment_issues:
            print(f"    Q{idx}: {q[:60]}... -> {gt[:40]}")
    else:
        print("  [OK] All questions have document support")

    print("\n--- Saving Cleaned Data ---")
    df_doc_clean.to_parquet(DOC_OUTPUT, index=False)
    print(f"  Saved: {DOC_OUTPUT}")
    df_q_clean.to_parquet(Q_OUTPUT, index=False)
    print(f"  Saved: {Q_OUTPUT}")

    print("\n--- Generating Report ---")
    report = generate_report(doc_stats, q_stats, alignment_issues)
    with open(REPORT_OUTPUT, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Saved: {REPORT_OUTPUT}")

    print("\n" + "=" * 60)
    print("  Cleaning Complete!")
    print("=" * 60)
    print(f"\n  Documents: {doc_stats['original_count']:,} -> {doc_stats['final_count']:,}")
    print(f"    - Mojibake fixed:    {doc_stats['mojibake_fixed']:,}")
    print(f"    - Noise cleaned:     {doc_stats['noise_cleaned']:,}")
    print(f"    - Duplicates removed: {doc_stats['duplicates_removed']:,}")
    print(f"    - Short docs removed: {doc_stats['short_docs_removed']:,}")
    print(f"\n  Questions: {q_stats['original_count']:,} -> {q_stats['final_count']:,}")
    print(f"    - S08_ fixed (Q):    {q_stats['s08_fixed_questions']:,}")
    print(f"    - S08_ fixed (GT):   {q_stats['s08_fixed_ground_truths']:,}")
    print(f"    - Yes/No normalized: {q_stats['yes_no_normalized']:,}")
    print(f"    - Questions reformatted: {q_stats['questions_reformatted']:,}")
    print(f"    - Garbage removed:   {q_stats['garbage_removed']:,}")
    print(f"    - Flagged unanswerable: {q_stats['unanswerable_flagged']:,}")


if __name__ == '__main__':
    main()