# Simple Explanation of the NLP Project

This project contains 3 NLP tasks.

Each task teaches the model a different NLP skill.

---

# 1) Sonnet Generation

## What is this task?

The model learns how to write poetry like Shakespeare.

We train the model on Shakespeare sonnets so it learns:
- Writing style
- Vocabulary
- Rhythm
- Poem structure

---

## Input

The beginning of a poem.

Example:

In loving thee thou know’st I am forsworn,

---

## Output

The model completes the poem.

Example:

But thou art twice forsworn, to me love swearing;

---

## How does it work?

We use a language model like GPT-2.

The model predicts the next word again and again.

Example:

Input:
I love

Prediction:
you

Then:
I love you

Prediction:
so

Then:
I love you so

and continues like this.

---

## Steps

1. Load Shakespeare dataset
2. Tokenize text
3. Fine-tune GPT-2
4. Generate poems
5. Evaluate generated text

---

# 2) Paraphrase Detection

## What is this task?

The model checks whether two sentences mean the same thing.

---

## Example 1

Sentence 1:
How do I learn Python?

Sentence 2:
What is the best way to study Python?

Output:
YES

Because both sentences have the same meaning.

---

## Example 2

Sentence 1:
How to cook pasta?

Sentence 2:
How to repair a car?

Output:
NO

Because they are completely different.

---

## Input

Two sentences.

---

## Output

YES or NO

---

## How does it work?

The transformer model reads both sentences and learns semantic meaning.

The model compares:
- Words
- Context
- Sentence meaning

instead of only exact word matching.

---

## Steps

1. Load QQP dataset
2. Clean text
3. Tokenize sentence pairs
4. Train BERT/RoBERTa
5. Predict similarity
6. Evaluate accuracy

---

# 3) Sentiment Classification

## What is this task?

The model detects emotions or opinions in text.

Usually:
- Positive
- Negative
- Neutral

---

## Example 1

Input:
This movie is amazing.

Output:
Positive

---

## Example 2

Input:
This product is terrible.

Output:
Negative

---

## Input

A review or sentence.

---

## Output

Sentiment label.

---

## How does it work?

The model learns emotional patterns in language.

Positive words:
- amazing
- beautiful
- excellent

Negative words:
- terrible
- bad
- boring

But transformers also understand context.

Example:

"This movie is not bad"

Traditional models may think:
bad → negative

But transformers understand:
not bad → positive meaning

---

## Steps

1. Load SST and CFIMDB datasets
2. Clean text
3. Tokenize sentences
4. Train classifier
5. Predict sentiment
6. Evaluate results

---

# Difference Between The 3 Tasks

| Task | Input | Output | NLP Type |
|---|---|---|---|
| Sonnet Generation | Beginning of poem | Generated poem | Text Generation |
| Paraphrase Detection | Two sentences | YES / NO | Similarity Classification |
| Sentiment Classification | Review sentence | Positive/Negative | Text Classification |

---

# Suggested Models

| Task | Suggested Model |
|---|---|
| Sonnet Generation | GPT-2 |
| Paraphrase Detection | BERT / RoBERTa |
| Sentiment Classification | DistilBERT / BERT |

---

# Possible Improvements

To make the project stronger:

- Use larger transformer models
- Better preprocessing
- Hyperparameter tuning
- Data augmentation
- DPO training
- Ensemble models
- Better sampling strategies

---

# Final Goal

The final goal is:
- Understand transformer NLP models
- Compare baseline vs improved methods
- Analyze results
- Build a research-style NLP project

---