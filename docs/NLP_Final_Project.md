# NLP Final Project

## Project Title
Improving NLP Tasks Using Transformer-Based Models

---

# Team Members
- Mohamed Abdelhady
- Student 2
- Student 3

---

# Overview

This project focuses on solving and improving three important Natural Language Processing (NLP) tasks using Transformer-based deep learning models.

The project includes:

1. Sonnet Generation
2. Paraphrase Detection
3. Sentiment Classification

We will compare baseline transformer models with improved approaches and analyze the performance on different datasets.

---

# Task 1 — Sonnet Generation

## Goal
Generate Shakespeare-style sonnets using a language model.

## Dataset
Shakespeare Sonnets Dataset

## Problem Type
Text Generation

## Input
A starting text prompt or first few lines of a sonnet.

Example Input:
In loving thee thou know’st I am forsworn,

## Output
The model generates the continuation of the poem.

Example Output:
But thou art twice forsworn, to me love swearing;
In act thy bed-vow broke, and new faith torn.

---

## Models

### Baseline
- GPT-2

### Proposed Improvement
Possible improvements:
- Fine-tuning larger transformer models
- Better decoding methods
- Direct Preference Optimization (DPO)
- Temperature tuning
- Top-p sampling

---

## Evaluation Metrics
- ChrF Score
- BLEU Score
- Human Evaluation

---

# Task 2 — Paraphrase Detection

## Goal
Determine whether two sentences have the same meaning.

## Dataset
QQP (Quora Question Pairs)

## Problem Type
Sentence Similarity / Binary Classification

## Input
Two sentences/questions.

Example Input:
Sentence 1: How can I learn programming?
Sentence 2: What is the best way to study coding?

## Output
YES → Same meaning

or

NO → Different meaning

---

## Models

### Baseline
- BERT
- GPT-2 Cloze Style Classification

### Proposed Improvement
Possible improvements:
- RoBERTa
- Better fine-tuning
- Data augmentation
- DPO training
- Contrastive learning

---

## Evaluation Metrics
- Accuracy
- F1-Score
- Precision
- Recall

---

# Task 3 — Sentiment Classification

## Goal
Classify text sentiment as positive or negative.

## Datasets
- SST (Stanford Sentiment Treebank)
- CFIMDB

## Problem Type
Text Classification

## Input
A movie or product review.

Example Input:
This movie was amazing and emotionally powerful.

## Output
Positive

---

## Another Example

Input:
The movie was boring and too long.

Output:
Negative

---

## Models

### Baseline
- DistilBERT
- BERT

### Proposed Improvement
Possible improvements:
- RoBERTa
- Ensemble methods
- Better preprocessing
- Attention visualization
- Hyperparameter tuning

---

## Evaluation Metrics
- Accuracy
- F1-Score
- Confusion Matrix

---

# General Workflow

## Step 1 — Data Collection
Download and preprocess datasets.

## Step 2 — Data Cleaning
- Remove unnecessary symbols
- Tokenization
- Lowercasing
- Padding and truncation

## Step 3 — Model Training
Train transformer models on each task.

## Step 4 — Evaluation
Compare baseline models with improved approaches.

## Step 5 — Analysis
Analyze:
- Errors
- Failure cases
- Accuracy improvements
- Generated text quality

---

# Expected Outcomes

We expect transformer-based models with improved training techniques to outperform the baseline methods across all tasks.

---

# Technologies

- Python
- PyTorch
- HuggingFace Transformers
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

# References

1. Stanford CS224N Projects
2. GPT-2 Paper
3. BERT Paper
4. RoBERTa Paper
5. DPO Paper

---