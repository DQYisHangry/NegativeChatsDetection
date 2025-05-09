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

CHEN Han. gamer's negative chat recognition(消极游戏聊天内容检测). https://kaggle.com/competitions/gamers-negative-chat-recognition, 2022. Kaggle.

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

  ·  I first removed duplicate messages to avoid model overfitting and ensured clean formatting. 

### 2. Feature modification:

  ·  Key words: To enhance the model's understanding of gaming-specific toxity, I curated a list of toxic keywords (e.g., 挂机, 送人头, etc.)
 
  ·  text_len: I set the text max length to 25, as each text in this data were quite short (about 18 characters per text on average)
 
  ·  excl_count, ques_count: I also added simple text-based featues like punctuation counts. Count of ! and ? as potential aggression markers


### 3. Cleaning & Tokenization:

  ·  I cleaned the text by removing non-Chinese characters using regex. 
 
  ·  For the baseline model tokenization, I used Jieba to segment Chinese text into words. 
  
  ·  For the final model, I used HuggingFace's BERT tokenizer. 



## Baseline Modeling


### Models:

I experimented with a few simple models to build intuition:

  ·  Logistic Regression + CountVectorizer
  
  ·  TF-IDF: Logistic Regression
  
  ·  MLPClassifier + Word2Vec sentence embedding: I used Tencen's pretrained word embeddings to generate sentence-level featues, but performance was poor (F1~0.53 for negative texts), so I abandoned this path. 

  ·  My guess on this poor performance are: 
  
       a. Mismacthed Domain: 
       The Tencent embeddings were trained on genral Chinese corpora like news, blogs and encylopedias, which differ significantly from informal, slang-heavy game chat. 
       As a result, many in-game phrases and toxic expressions were either poorly represented or missing from the embedding space. 
       
       b. Short&Spares Texts: 
       Game chat messages are extremely short (often under 15 characters), limiting the contexutal richness that Word2Vec relies on. 
       With only a few words per message, sentence-level averaging of embeddings tends to dilute meaningful distinctions between classes. 

While these approches were lightweight and easy to implement, they quickly hit a performance ceilinf - particularly in capturing nuanced or implicit negativity in short, slang-heavy game chat messages. 


### Evaluation. The results of the baseline model:

  ·  Metric: F1-score (focus on class 1)

  ·  Also monitored precision and recall due to imbalanced classes
![1746689923014](https://github.com/user-attachments/assets/652f396b-f7d8-485f-a2ea-5ff7d53c25d0)




### Training Environment: 
Local Jupyter Notebook


## BERT Fine-Tuning (Main Model)
The base line models les me to fine-tune a pre-trained Chinese BERT model. Unlike static embeddings, BERT provides context-aware representations that are much better suited to short, informal language. Fine-tuning gave a significant boost in F1 and allowed the model to better distinguish subtle toxicity pattern. 


### Model: 
The bert-base-Chinese model is a version of BERT pretrained by Google on large-scale Chinese text (including Chinese Wikipedia) and published via Hugging Face's model hub: 
https://huggingface.co/bert-base-chinese


### Preprocessing: 
Custom jieba tokenizer + HuggingFace tokenizer


### Trainer: 
HuggingFace Trainer


### Hyperparameters:

I optimized the max sequence length (set to 25), batch size (32/64), learning rate, and epoch count(5). 
I also used stratified train/val splits (by label and keyword) to ensure consistent class distribution. 

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

## Results:
This model achieved and F1 score of ~0.62 on the validation set. Still not perfect but it's a much better improvement from the baseline model. 

# Final Model: BERT + LightGBM Ensemble

To further boost performance, I implemented a simple ensemble:
  ·  I extracted logits (softmax probabilities) from the fine-tuned BERT model on the validation set. 
  ·  I combined them with the keyword_flag as features to train a LightGB< classifier. 
  ·  This ensenble pushed the F1-score to ~0,65, with improved precision and recall. 


# Evaluation Summary 

| Model                      | Precision | Recall | F1 Score |
|---------------------------|-----------|--------|----------|
| MLP + Word2Vec            | 0.53      | 0.54   | 0.53     |
| BERT                      | 0.70      | 0.55   | 0.62     |
| BERT + LightGBM (Final)   | 0.75      | 0.57   | 0.65     |


## Final Thoughts
### This project has been deeply rewarding exploration into the intersection of NLP and online behavior moderation. From experimenting with tradtional vectorization methods to fine-tuning state-of-the-art transformer models, I experienced firsthand the trade-offs between speed, interpretability, and performance. While my early attempts with Word2vec and logistic regression laid a solid foundation, it was the shift to BERT fine-tuning - and eventully to BERT + LightGBM meta-classifier - that brought significant gains in F1-score. Along the way, I strenthened my ability to troubleshoot, optimize, and iterate independently. 
