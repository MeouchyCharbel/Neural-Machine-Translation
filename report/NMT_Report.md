# Neural Machine Translation: English-to-French Translation Using LSTM and Transformer Models

## 1. Introduction

Neural Machine Translation (NMT) uses artificial neural networks to translate text from one language into another. Unlike dictionary-based translation, neural translation models learn linguistic patterns from sentence pairs and generate translations based on the input sequence.

This project investigates English-to-French translation using several approaches:

1. Dictionary-based translation
2. TF-IDF retrieval-based translation
3. LSTM encoder–decoder without attention
4. LSTM encoder–decoder with attention
5. Pretrained MarianMT Transformer

The dictionary and TF-IDF methods serve as introductory baselines. The final experiment focuses on comparing the three neural models.

**Research question:** How does adding attention affect an LSTM translation model, and how do both LSTM variants compare with a pretrained Transformer?

The final comparison uses the same 2,000 held-out test sentences for all three neural models.

### 1.1 Project Objectives

- Prepare and preprocess an English–French parallel dataset.
- Implement dictionary and TF-IDF translation baselines.
- Evaluate an existing pretrained Transformer.
- Compare greedy decoding with beam search.
- Build and train an LSTM without attention.
- Build and train an LSTM with attention.
- Compare the neural models using identical test sentences and references.
- Measure translation quality using BLEU and chrF.
- Examine example translations and discuss model limitations.

---

## 2. Dataset and Preprocessing

### 2.1 Dataset

The project uses an English–French parallel corpus obtained from the TensorFlow-hosted `fra-eng.zip` archive.

Each example contains an English sentence and its corresponding French translation.

The original dataset contained **167,130 sentence pairs**.

After preprocessing, **166,340 sentence pairs** remained.

### 2.2 Preprocessing

The preprocessing pipeline includes:

- Removing missing and empty sentences.
- Removing duplicate English–French sentence pairs.
- Keeping English sentences containing at most 15 words.
- Normalizing text for overlap detection.
- Grouping related sentences to reduce data leakage.
- Separating the corpus into training, validation, and test partitions.

### 2.3 Dataset Splitting

A group-aware splitting strategy was used to reduce overlap between partitions.

| Partition | Sentence pairs |
|---|---:|
| Training | 133,017 |
| Validation | 16,616 |
| Test | 16,707 |
| **Total** | **166,340** |

The implemented exact and normalized overlap checks passed.

For the final experiment, the following subsets were selected:

| Purpose | Sentence pairs |
|---|---:|
| LSTM training | 50,000 |
| Validation monitoring | 1,000 |
| Final testing | 2,000 |

Both LSTMs were trained using the same 50,000 training pairs. The 1,000 validation-monitor examples were used to select the best trained checkpoints.

**The 2,000 held-out test examples were not used to train the models or select checkpoints.**

This separation allows the final comparison to assess translation performance on previously unseen examples from the project's data split.

---

## 3. Translation Methods

### 3.1 Dictionary Baseline

The dictionary baseline translates English sentences word by word using a small predefined English–French dictionary.

This approach is computationally simple but has several limitations:

- Limited vocabulary.
- No contextual understanding.
- No grammatical modeling.
- Difficulty translating unknown expressions.

The dictionary provides an introductory baseline rather than a competitive translation system.

### 3.2 TF-IDF Retrieval Baseline

The TF-IDF approach represents English sentences as numerical vectors based on word importance.

For a new English sentence, the system:

1. Converts the sentence into a TF-IDF vector.
2. Calculates cosine similarity against training sentences.
3. Selects the most similar English sentence.
4. Returns its associated French translation.

The retrieval index contains **92,892 unique English training sentences**.

Unlike neural machine translation, this method retrieves an existing translation rather than generating new French tokens.

### 3.3 LSTM Without Attention

LSTM stands for Long Short-Term Memory. It is a recurrent neural network architecture designed to model sequential information.

The first neural model uses:

- A bidirectional LSTM encoder to process English input tokens.
- Trainable English and French token embeddings.
- An LSTM decoder that generates French tokens sequentially.
- Encoder final states to initialize the decoder.

The decoder produces translations without directly attending to all encoder outputs at every generation step.

This creates a useful comparison model for studying the contribution of attention.

### 3.4 LSTM With Attention

The second LSTM uses the same general encoder–decoder structure, training dataset, tokenization settings, and training hyperparameters.

However, it adds a **four-head cross-attention mechanism**.

Attention allows the decoder to use information from different positions in the encoded English sentence during translation.

Instead of relying only on the encoded final states, the decoder can incorporate additional information from the sequence of encoder outputs.

The attention-based model has additional trainable parameters, so this comparison does not isolate attention independently of model capacity.

### 3.5 Pretrained Transformer — MarianMT

The pretrained model used in this project is:

`Helsinki-NLP/opus-mt-en-fr`

MarianMT is an encoder–decoder Transformer for English-to-French translation.

Unlike an LSTM, a Transformer relies primarily on attention mechanisms rather than recurrent processing to model relationships between tokens.

In this project, MarianMT was used **without additional fine-tuning**.

It had already been pretrained using translation data outside this project, whereas the two
