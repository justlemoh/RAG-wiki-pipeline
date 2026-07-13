"""
Evaluation Module for RAG Wiki Pipeline
=======================================
Benchmarks retrieval performance on all free-text questions and runs
end-to-end RAG evaluation on a subset of questions using the Gemini API.
"""

import os
import random
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import sys
# Add project root to sys.path so imports work when running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retrieve import retrieve
from rag.rag_query import run_rag_pipeline

# Load environment variables from .env file
load_dotenv()

QUESTIONS_PATH = 'data/clean/questions_cleaned.parquet'
SAMPLE_SIZE = 40  # Sample size for end-to-end LLM generation evaluation


def clean_text_for_eval(text):
    """Normalize text for substring comparison."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return " ".join(text.split())


def evaluate_retrieval(df_q, k_values=[1, 3, 5]):
    """
    Evaluate retrieval on Free Text questions.
    Checks if the normalized ground truth answer is present as a substring
    in the retrieved document chunks.

    Runs two passes:
      1. Without reranking (pure vector similarity from Qdrant)
      2. With cross-encoder reranking
    Prints a side-by-side comparison table.
    """
    print("\n--- Evaluating Retrieval Module (Free-Text Questions only) ---")

    # Filter to free-text questions (excluding yes/no answers as they are too common)
    df_free = df_q[~df_q['ground_truth'].isin(['yes', 'no'])].copy()
    total_free = len(df_free)
    print(f"Found {total_free} free-text questions for retrieval benchmarking.")

    if total_free == 0:
        print("No free-text questions found for evaluation.")
        return {}

    # Pre-cache the reranker model before the evaluation loop so that its
    # first-use download/initialization cost is not counted per-query.
    from rag.retrieve import get_reranker
    print("Pre-loading reranker model (downloads on first run)...")
    get_reranker()
    print("  Reranker ready.\n")

    all_metrics = {}

    for use_rerank in [False, True]:
        mode = "With Reranking" if use_rerank else "Without Reranking (vector only)"
        hits = {k: 0 for k in k_values}

        for _, row in tqdm(df_free.iterrows(), total=total_free, desc=f"Retrieving [{mode}]"):
            q = row['question']
            gt = clean_text_for_eval(row['ground_truth'])

            if not gt:
                continue

            max_k = max(k_values)
            results = retrieve(q, k=max_k, rerank=use_rerank)

            for k in k_values:
                retrieved_texts = [clean_text_for_eval(r['document']) for r in results[:k]]
                if any(gt in text for text in retrieved_texts):
                    hits[k] += 1

        metrics = {}
        for k in k_values:
            recall = hits[k] / total_free
            metrics[f"Recall@{k}"] = recall
        all_metrics[mode] = metrics

    # Print comparison table
    print("\nRetrieval Recall (Hit Rate) Comparison:")
    col_w = 35
    header = f"  {'Metric':<12}" + "".join(f"  {m:<{col_w}}" for m in all_metrics)
    print(header)
    print("  " + "-" * (12 + (col_w + 2) * len(all_metrics)))
    for k in k_values:
        row_str = f"  {'Recall@' + str(k):<12}"
        for mode, metrics in all_metrics.items():
            val = metrics[f"Recall@{k}"]
            cell = f"{val:.2%} ({int(val * total_free)}/{total_free})"
            row_str += f"  {cell:<{col_w}}"
        print(row_str)

    # Return metrics for the reranked mode (primary result)
    return all_metrics.get("With Reranking", {})



def calculate_f1_score(prediction, ground_truth):
    """Calculate word-level F1 score between prediction and ground truth."""
    pred_tokens = clean_text_for_eval(prediction).split()
    gt_tokens = clean_text_for_eval(ground_truth).split()
    
    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0
        
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
        
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


from collections import Counter

def evaluate_generation(df_q, sample_size=SAMPLE_SIZE):
    """
    Run end-to-end RAG evaluation on a random sample of the questions.
    Checks accuracy of yes/no answers and computes F1 score for free-text answers.
    """
    print(f"\n--- Evaluating RAG Generation Module (Random Sample of {sample_size} questions) ---")
    
    # Sample a mix of binary and free text if possible
    df_yes_no = df_q[df_q['ground_truth'].isin(['yes', 'no'])]
    df_free = df_q[~df_q['ground_truth'].isin(['yes', 'no'])]
    
    half_size = sample_size // 2
    sample_yes_no = df_yes_no.sample(min(half_size, len(df_yes_no)), random_state=42)
    sample_free = df_free.sample(min(sample_size - len(sample_yes_no), len(df_free)), random_state=42)
    
    df_sample = pd.concat([sample_yes_no, sample_free]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    results = []
    yes_no_correct = 0
    yes_no_total = 0
    free_f1_scores = []

    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Generating Answers"):
        q = row['question']
        gt = row['ground_truth']
        
        # Run pipeline
        pred, chunks = run_rag_pipeline(q, k=3)
        
        is_binary = gt in ['yes', 'no']
        correct = False
        f1 = 0.0
        
        if is_binary:
            yes_no_total += 1
            pred_clean = pred.lower().strip().rstrip('.')
            # Check if answer starts with or matches gt
            if pred_clean == gt or pred_clean.startswith(gt):
                correct = True
                yes_no_correct += 1
            elif gt == 'yes' and ('yes' in pred_clean and 'no' not in pred_clean):
                correct = True
                yes_no_correct += 1
            elif gt == 'no' and ('no' in pred_clean and 'yes' not in pred_clean):
                correct = True
                yes_no_correct += 1
        else:
            f1 = calculate_f1_score(pred, gt)
            free_f1_scores.append(f1)
            # Mark as correct if F1 score > 0.5 (loose match) or gt is substring of pred
            gt_clean = clean_text_for_eval(gt)
            pred_clean = clean_text_for_eval(pred)
            if gt_clean in pred_clean or f1 >= 0.5:
                correct = True
                
        results.append({
            'question': q,
            'ground_truth': gt,
            'prediction': pred,
            'is_binary': is_binary,
            'correct': correct,
            'f1_score': f1 if not is_binary else None
        })

    # Summary
    print("\nGeneration Evaluation Results:")
    
    yes_no_acc = yes_no_correct / yes_no_total if yes_no_total > 0 else 0.0
    mean_f1 = np.mean(free_f1_scores) if free_f1_scores else 0.0
    
    if yes_no_total > 0:
        print(f"  Yes/No Accuracy: {yes_no_acc:.2%} ({yes_no_correct}/{yes_no_total})")
    if free_f1_scores:
        print(f"  Free-Text Mean F1-Score: {mean_f1:.2%}")
        
    # Print a few sample predictions
    print("\n--- Sample Predictions ---")
    for i, res in enumerate(results[:5], 1):
        print(f"\n{i}. Question: {res['question']}")
        print(f"   Ground Truth: '{res['ground_truth']}'")
        print(f"   RAG Prediction: '{res['prediction']}'")
        print(f"   Correct: {res['correct']}")
        
    return {
        'YesNo_Accuracy': yes_no_acc,
        'FreeText_Mean_F1': mean_f1
    }


def main():
    if not os.path.exists(QUESTIONS_PATH):
        print(f"Error: Cleaned questions file not found at {QUESTIONS_PATH}. Run clean_data.py first.")
        return

    df_q = pd.read_parquet(QUESTIONS_PATH)
    print(f"Loaded {len(df_q)} cleaned questions.")

    # 1. Evaluate Retrieval
    evaluate_retrieval(df_q, k_values=[1, 3, 5])

    # 2. Evaluate Generation (End-to-End)
    evaluate_generation(df_q, sample_size=SAMPLE_SIZE)


if __name__ == '__main__':
    main()
