# Gamers' Negative Chat Recognition - Modeling Experiment

## Overview

This project addresses the growing concern of negative player behavior in online games, such as toxic language, threats, intentional feeding, or AFK (away from keyboard) behavior. Our goal is to detect negative chat messages in a Chinese-language dataset using NLP and supervised machine learning.

---

## Problem Statement

Online multiplayer games often suffer from disruptive player behavior, such as verbal abuse, intentional feeding, AFK (away from keyboard), and rage-quitting. These behaviors degrade the gaming experience. Our objective is to:
Classify each chat message as either Normal (label=0) or Negative (label=1).

---

## Dataset

### Source: Provided CSV file from competition organizers

### Fields: 
qid, text, label

### Size:
~60,000 rows

### Distribution:

Normal (label = 0): ~62%

Negative (label = 1): ~38%

---

## Data Acquisition

### To load the dataset in your notebook:
<pre>import pandas as pd 

df = pd.read_csv("/content/data/train.csv")  </pre>

---

## Preprocessing Steps

### 1. Deduplication: Removed duplicated text entries


### 2. Feature modification:

  ·  keyword_flag: Whether the text contains known negative words (e.g., 挂机, 送人头)
 
  ·  text_len: Length of message
 
  ·  excl_count, ques_count: Count of ! and ? as potential aggression markers


### 3. Cleaning & Tokenization:

  ·  Regex to remove non-Chinese characters
 
  ·  Tokenized using jieba


### 4. Initial Word2Vec Attempt (Baseline):

  ·  Used jieba for preprocessing and tokenization
 
  ·  Applied Tencent's pretrained Chinese Word2Vec embeddings
 
 · Trained an MLPClassifier using sentence-level averaged embeddings
 

## Baseline Modeling

### Models:

  ·  Logistic Regression + CountVectorizer

  ·  MLPClassifier + Word2Vec sentence embedding

  ·  LightGBM + BERT logits + keyword features
  

### Evaluation:

  ·  Metric: F1-score (focus on class 1)

  ·  Also monitored precision and recall due to imbalanced classes
![1746689923014](https://github.com/user-attachments/assets/652f396b-f7d8-485f-a2ea-5ff7d53c25d0)



### Training Environment: 
Local Jupyter Notebook


## BERT Fine-Tuning (Main Model)

### Model: 
bert-base-chinese


### Preprocessing: 
Custom jieba tokenizer + HuggingFace tokenizer


### Trainer: 
HuggingFace Trainer


### Hyperparameters:


  ·  Max sequence length: 25

  ·  Epochs: 5

  ·  Batch size: 32 (train) / 64 (eval)

  ·  Learning rate: 2e-5

  ·  Weight decay: 0.01

  ·  Metric for model selection: eval_f1

## Training Environment: 

### Platform: 
Google Colab Pro

### GPU: 
T4 (16GB)

# Final Model: BERT + LightGBM Ensemble

I used BERT's prediction logits as features, combined with keyword_flag, and trained a LightGBM classifier to improve recall and F1.

## Evaluation Summary

| Model                      | Precision | Recall | F1 Score |
|---------------------------|-----------|--------|----------|
| MLP + Word2Vec            | 0.53      | 0.54   | 0.53     |
| BERT                      | 0.70      | 0.55   | 0.62     |
| BERT + LightGBM (Final)   | 0.75      | 0.57   | 0.65     |


