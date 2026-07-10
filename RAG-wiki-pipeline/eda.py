import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Create directory for plots if it doesn't exist
os.makedirs('plots', exist_ok=True)

# 1. Load data
print("Loading data...")
df_doc = pd.read_parquet('data/documents.parquet')
df_q = pd.read_parquet('data/questions.parquet')

# 2. Basic dataset summaries
doc_info = {
    'total_rows': len(df_doc),
    'null_values': df_doc['document'].isnull().sum(),
    'duplicate_values': df_doc['document'].duplicated().sum(),
}

q_info = {
    'total_rows': len(df_q),
    'null_values_q': df_q['question'].isnull().sum(),
    'null_values_gt': df_q['ground_truth'].isnull().sum(),
    'duplicate_questions': df_q['question'].duplicated().sum(),
}

# 3. Calculate text length statistics
print("Analyzing text lengths...")
# Documents length
df_doc['char_len'] = df_doc['document'].str.len()
df_doc['word_count'] = df_doc['document'].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)

# Questions length
df_q['char_len'] = df_q['question'].str.len()
df_q['word_count'] = df_q['question'].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)

doc_stats = df_doc[['char_len', 'word_count']].describe()
q_stats = df_q[['char_len', 'word_count']].describe()

# 4. Plot distributions
print("Generating plots...")

# Plot 1: Document Length Distribution (Word Count)
plt.figure(figsize=(10, 6))
sns.histplot(df_doc['word_count'], bins=30, kde=True, color='#2b5c8f')
plt.title('Document Word Count Distribution', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Word Count', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig('plots/document_word_count_dist.png', dpi=300)
plt.close()

# Plot 2: Question Length Distribution (Word Count)
plt.figure(figsize=(10, 6))
sns.histplot(df_q['word_count'], bins=20, kde=True, color='#d95f02')
plt.title('Question Word Count Distribution', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Word Count', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig('plots/question_word_count_dist.png', dpi=300)
plt.close()

# Plot 3: Ground Truth Distribution (Yes vs No vs Free Text)
plt.figure(figsize=(8, 6))

def categorize_gt(val):
    if not isinstance(val, str):
        return "Unknown/Null"
    val_clean = val.lower().strip().rstrip('.')
    if val_clean == 'yes':
        return 'Yes'
    elif val_clean == 'no':
        return 'No'
    else:
        return 'Free Text'

df_q['gt_category'] = df_q['ground_truth'].apply(categorize_gt)
gt_counts = df_q['gt_category'].value_counts()
order = [x for x in ['Yes', 'No', 'Free Text'] if x in gt_counts.index]
gt_counts = gt_counts.reindex(order)

sns.barplot(x=gt_counts.index, y=gt_counts.values, hue=gt_counts.index, palette={'Yes': '#2ca02c', 'No': '#d62728', 'Free Text': '#1f77b4'}, legend=False)
plt.title('Ground Truth Answer Categories', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Answer Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
# Add percentage labels on top of bars
total = len(df_q)
for i, (cat, count) in enumerate(gt_counts.items()):
    pct = 100 * count / total
    plt.text(i, count + total * 0.01, f'{count} ({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/ground_truth_dist.png', dpi=300)
plt.close()

# Helper for N-grams
STOP_WORDS = set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'cannot',
    'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further',
    'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his',
    'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'me', 'more', 'most', 'my', 'myself',
    'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
    'ourselves', 'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some', 'such', 'than', 'that',
    'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
    'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when',
    'where', 'which', 'while', 'who', 'whom', 'why', 'with', 'would', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'is', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must'
])

def get_top_ngrams(corpus, n=1, top_k=15):
    words = []
    for text in corpus:
        if not isinstance(text, str):
            continue
        # Clean text
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
        
        if n == 1:
            words.extend(tokens)
        else:
            # Generate n-grams
            ngrams = zip(*[tokens[i:] for i in range(n)])
            words.extend([" ".join(ngram) for ngram in ngrams])
            
    return Counter(words).most_common(top_k)

print("Calculating top words and bigrams...")
top_doc_unigrams = get_top_ngrams(df_doc['document'], n=1, top_k=15)
top_doc_bigrams = get_top_ngrams(df_doc['document'], n=2, top_k=15)

top_q_unigrams = get_top_ngrams(df_q['question'], n=1, top_k=15)
top_q_bigrams = get_top_ngrams(df_q['question'], n=2, top_k=15)

# Plot 4: Top 15 Unigrams in Documents
plt.figure(figsize=(10, 6))
words, counts = zip(*top_doc_unigrams)
sns.barplot(x=list(counts), y=list(words), color='#1f77b4')
plt.title('Top 15 Most Common Words in Documents (Excl. Stop Words)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Word', fontsize=12)
plt.tight_layout()
plt.savefig('plots/top_words_docs.png', dpi=300)
plt.close()

# Plot 5: Top 15 Bigrams in Documents
plt.figure(figsize=(10, 6))
words, counts = zip(*top_doc_bigrams)
sns.barplot(x=list(counts), y=list(words), color='#17becf')
plt.title('Top 15 Most Common Bigrams in Documents (Excl. Stop Words)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Bigram', fontsize=12)
plt.tight_layout()
plt.savefig('plots/top_bigrams_docs.png', dpi=300)
plt.close()

# Plot 6: Top 15 Unigrams in Questions
plt.figure(figsize=(10, 6))
words, counts = zip(*top_q_unigrams)
sns.barplot(x=list(counts), y=list(words), color='#ff7f0e')
plt.title('Top 15 Most Common Words in Questions (Excl. Stop Words)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Word', fontsize=12)
plt.tight_layout()
plt.savefig('plots/top_words_questions.png', dpi=300)
plt.close()

# Plot 7: Top 15 Bigrams in Questions
plt.figure(figsize=(10, 6))
words, counts = zip(*top_q_bigrams)
sns.barplot(x=list(counts), y=list(words), color='#bcbd22')
plt.title('Top 15 Most Common Bigrams in Questions (Excl. Stop Words)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Bigram', fontsize=12)
plt.tight_layout()
plt.savefig('plots/top_bigrams_questions.png', dpi=300)
plt.close()

# Generate Markdown Report
print("Generating Markdown report...")
report_content = f"""# Exploratory Data Analysis (EDA) Report

This report summarizes the findings of the Exploratory Data Analysis performed on the `rag-wiki-pipeline` datasets.

## 1. Dataset Overview

### Documents Dataset (`data/documents.parquet`)
- **Total Records:** {doc_info['total_rows']:,}
- **Missing Values:** {doc_info['null_values']}
- **Duplicate Documents:** {doc_info['duplicate_values']}

### Questions Dataset (`data/questions.parquet`)
- **Total Records:** {q_info['total_rows']:,}
- **Missing Questions:** {q_info['null_values_q']}
- **Missing Ground Truths:** {q_info['null_values_gt']}
- **Duplicate Questions:** {q_info['duplicate_questions']}

---

## 2. Text Statistics & Distributions

### Documents (Wikipedia Passages)
- **Average Character Length:** {doc_stats.loc['mean', 'char_len']:.1f} characters (min: {doc_stats.loc['min', 'char_len']:.0f}, max: {doc_stats.loc['max', 'char_len']:.0f})
- **Average Word Count:** {doc_stats.loc['mean', 'word_count']:.1f} words (min: {doc_stats.loc['min', 'word_count']:.0f}, max: {doc_stats.loc['max', 'word_count']:.0f})

*Document Word Count Distribution is saved as `plots/document_word_count_dist.png`*

### Questions
- **Average Character Length:** {q_stats.loc['mean', 'char_len']:.1f} characters (min: {q_stats.loc['min', 'char_len']:.0f}, max: {q_stats.loc['max', 'char_len']:.0f})
- **Average Word Count:** {q_stats.loc['mean', 'word_count']:.1f} words (min: {q_stats.loc['min', 'word_count']:.0f}, max: {q_stats.loc['max', 'word_count']:.0f})

*Question Word Count Distribution is saved as `plots/question_word_count_dist.png`*

---

## 3. Class Balance (Ground Truth Answers)

The questions dataset contains a mix of binary (Yes/No) questions and free-text (factual Q&A) questions. Here is the distribution of the answer categories:

- **Yes:** {gt_counts.get('Yes', 0)} ({100 * gt_counts.get('Yes', 0) / total:.1f}%)
- **No:** {gt_counts.get('No', 0)} ({100 * gt_counts.get('No', 0) / total:.1f}%)
- **Free Text (factual/detailed):** {gt_counts.get('Free Text', 0)} ({100 * gt_counts.get('Free Text', 0) / total:.1f}%)

This shows that the evaluation dataset tests both binary decision-making and precise factual extraction capabilities.

*Ground Truth Class Distribution is saved as `plots/ground_truth_dist.png`*

---

## 4. Top Unigrams and Bigrams (Word Frequency)

### Documents
#### Top 10 Words:
{chr(10).join([f"- **{word}**: {count} times" for word, count in top_doc_unigrams[:10]])}

#### Top 10 Bigrams:
{chr(10).join([f"- **{word}**: {count} times" for word, count in top_doc_bigrams[:10]])}

### Questions
#### Top 10 Words:
{chr(10).join([f"- **{word}**: {count} times" for word, count in top_q_unigrams[:10]])}

#### Top 10 Bigrams:
{chr(10).join([f"- **{word}**: {count} times" for word, count in top_q_bigrams[:10]])}

---

## Plots Generated:
1. `plots/document_word_count_dist.png`
2. `plots/question_word_count_dist.png`
3. `plots/ground_truth_dist.png`
4. `plots/top_words_docs.png`
5. `plots/top_bigrams_docs.png`
6. `plots/top_words_questions.png`
7. `plots/top_bigrams_questions.png`
"""

with open('EDA_Report.md', 'w', encoding='utf-8') as f:
    f.write(report_content)

print("EDA analysis completed. Files saved successfully.")
