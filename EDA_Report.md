# Exploratory Data Analysis (EDA) Report

This report summarizes the findings of the Exploratory Data Analysis performed on the `rag-wiki-pipeline` datasets.

## 1. Dataset Overview

### Documents Dataset (`data/documents.parquet`)
- **Total Records:** 3,200
- **Missing Values:** 0
- **Duplicate Documents:** 4

### Questions Dataset (`data/questions.parquet`)
- **Total Records:** 918
- **Missing Questions:** 0
- **Missing Ground Truths:** 0
- **Duplicate Questions:** 0

---

## 2. Text Statistics & Distributions

### Documents (Wikipedia Passages)
- **Average Character Length:** 389.8 characters (min: 1, max: 2515)
- **Average Word Count:** 62.1 words (min: 1, max: 425)

*Document Word Count Distribution is saved as `plots/document_word_count_dist.png`*

### Questions
- **Average Character Length:** 53.1 characters (min: 4, max: 252)
- **Average Word Count:** 9.1 words (min: 1, max: 48)

*Question Word Count Distribution is saved as `plots/question_word_count_dist.png`*

---

## 3. Class Balance (Ground Truth Answers)

The questions dataset contains a mix of binary (Yes/No) questions and free-text (factual Q&A) questions. Here is the distribution of the answer categories:

- **Yes:** 333 (36.3%)
- **No:** 75 (8.2%)
- **Free Text (factual/detailed):** 510 (55.6%)

This shows that the evaluation dataset tests both binary decision-making and precise factual extraction capabilities.

*Ground Truth Class Distribution is saved as `plots/ground_truth_dist.png`*

---

## 4. Top Unigrams and Bigrams (Word Frequency)

### Documents
#### Top 10 Words:
- **also**: 458 times
- **president**: 426 times
- **one**: 423 times
- **new**: 414 times
- **first**: 392 times
- **war**: 329 times
- **roosevelt**: 307 times
- **world**: 278 times
- **two**: 277 times
- **states**: 276 times

#### Top 10 Bigrams:
- **united states**: 185 times
- **new york**: 157 times
- **white house**: 70 times
- **polar bears**: 68 times
- **theodore roosevelt**: 61 times
- **nikola tesla**: 55 times
- **polar bear**: 53 times
- **world war**: 45 times
- **john adams**: 42 times
- **vice president**: 39 times

### Questions
#### Top 10 Words:
- **many**: 32 times
- **president**: 30 times
- **otters**: 25 times
- **turtles**: 25 times
- **born**: 24 times
- **species**: 23 times
- **monroe**: 23 times
- **largest**: 22 times
- **ford**: 22 times
- **romania**: 22 times

#### Top 10 Bigrams:
- **james monroe**: 15 times
- **polar bear**: 11 times
- **grover cleveland**: 10 times
- **united states**: 9 times
- **president united**: 8 times
- **polar bears**: 8 times
- **country world**: 7 times
- **new york**: 7 times
- **john adams**: 7 times
- **millard fillmore**: 7 times

---

## Plots Generated:
1. `plots/document_word_count_dist.png`
2. `plots/question_word_count_dist.png`
3. `plots/ground_truth_dist.png`
4. `plots/top_words_docs.png`
5. `plots/top_bigrams_docs.png`
6. `plots/top_words_questions.png`
7. `plots/top_bigrams_questions.png`
