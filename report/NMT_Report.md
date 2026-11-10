# Neural Machine Translation: English-to-French Translation Using LSTM and Transformer Models

## 1. Introduction

Neural Machine Translation (NMT) uses artificial neural networks to translate text from one language into another. Unlike simple word-by-word translation, neural models learn relationships between words and generate translations based on the source sentence.

This project investigates English-to-French translation by comparing four approaches:

1. Dictionary-based translation
2. TF-IDF retrieval-based translation
3. LSTM encoder–decoder with attention
4. Pretrained Transformer (MarianMT)

The main objective is to evaluate their translation quality using BLEU and chrF scores on the same validation sentences.

### 1.1 Project Objectives

- Prepare and preprocess English–French sentence pairs.
- Implement dictionary and TF-IDF baselines.
- Evaluate a pretrained Transformer.
- Compare greedy decoding and beam search.
- Build and train an attention-based LSTM translator.
- Evaluate all models using consistent translation metrics.
- Analyze the strengths and limitations of each approach.

## 2. Dataset and Preprocessing

### 2.1 Dataset

The project uses an English–French parallel corpus obtained from the TensorFlow-hosted `fra-eng.zip` archive.

Each observation contains an English sentence and its corresponding French translation.

The original dataset contained **167,130 sentence pairs**.

After cleaning, **166,340 sentence pairs** remained.

### 2.2 Preprocessing

The preprocessing procedure included:

- Removing missing values and empty sentences.
- Removing duplicate English–French sentence pairs.
- Keeping English sentences containing at most 15 words.
- Normalizing sentences for overlap detection.
- Grouping related sentence pairs to reduce data leakage.

### 2.3 Dataset Splitting

A group-aware splitting procedure was used to reduce overlap between the training, validation, and test partitions.

| Partition | Sentence pairs |
|---|---:|
| Training | 133,017 |
| Validation | 16,616 |
| Test | 16,707 |

The implemented exact and normalized overlap checks passed.

For the LSTM experiment, **50,000 training pairs** were selected from the training partition, while **1,000 validation examples** were used to monitor training.

The final comparison used **500 identical validation sentences**, with the same French references as Notebook 02.

## 3. Translation Methods

### 3.1 Dictionary Baseline

The dictionary baseline translates English sentences word by word using a small predefined bilingual dictionary.

Unknown words remain untranslated.

This approach is computationally simple but does not account for sentence context, grammatical agreement, or complex linguistic relationships.

### 3.2 TF-IDF Retrieval Baseline

The TF-IDF baseline uses Term Frequency–Inverse Document Frequency to represent English sentences as numerical vectors.

For each input sentence, cosine similarity identifies the closest English sentence from the training collection.

The corresponding stored French sentence is returned as the translation.

The system indexed **92,892 unique English training sentences**.

Unlike neural translation models, TF-IDF retrieval does not generate a new translation token by token.

### 3.3 LSTM Encoder–Decoder with Attention

The LSTM model was developed and trained from scratch.

Its architecture consists of:

**Encoder:** A bidirectional LSTM processes the input English sequence and produces contextual hidden representations.

**Decoder:** An LSTM generates the French translation sequentially.

**Attention mechanism:** A multi-head cross-attention layer allows the decoder to incorporate information from the encoder's output representations.

**Embeddings:** English and French subword tokens are represented using trainable embedding vectors.

SentencePiece tokenization was used for both languages, with separate vocabularies of 6,000 entries.

During training, teacher forcing provides the reference French tokens as decoder inputs.

During inference, the model generates translations autoregressively without using reference translations.

### 3.4 Pretrained Transformer

The pretrained Transformer used in the project was:

`Helsinki-NLP/opus-mt-en-fr`

This MarianMT model uses a Transformer encoder–decoder architecture based on attention mechanisms.

The Transformer was used without additional fine-tuning in the revised project.

Its previous evaluation predictions from Notebook 02 were reused for the shared validation comparison.

## 4. LSTM Training

### 4.1 Training Configuration

| Parameter | Configuration |
|---|---|
| Training examples | 50,000 |
| Validation-monitor examples | 1,000 |
| Maximum epochs | 5 |
| Batch size | 64 |
| Embedding dimension | 256 |
| Hidden dimension | 256 |
| Vocabulary size | 6,000 per language |
| Maximum sequence length | 48 tokens |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Loss function | Cross-entropy |
| Early-stopping patience | 2 epochs |
| Trainable parameters | 7,466,352 |

Gradient clipping was used during training.

The model was trained using a CUDA-enabled GPU.

### 4.2 Training Results

| Epoch | Training loss | Validation-monitor loss |
|---|---:|---:|
| 1 | 3.3145 | 2.3729 |
| 2 | 1.8107 | 1.9128 |
| 3 | 1.3092 | 1.7792 |
| 4 | 1.0170 | 1.7374 |
| 5 | 0.8113 | 1.7569 |

Training loss decreased throughout the experiment.

The monitoring loss improved until epoch 4 and increased slightly at epoch 5.

Therefore, the model weights from **epoch 4** were selected for the final evaluation.

The best monitoring loss was **1.7374**.

Early stopping was configured, but training completed all five planned epochs.

## 5. Evaluation Metrics

### 5.1 BLEU

BLEU (Bilingual Evaluation Understudy) measures the overlap between generated translations and reference translations using word n-grams.

It also includes a brevity penalty.

Higher BLEU scores generally indicate greater similarity to the reference translations.

### 5.2 chrF

chrF evaluates translation quality using character n-gram precision and recall.

It can capture similarities between words that share morphological features.

Both BLEU and chrF are reference-based metrics. Neither guarantees that a translation is semantically correct.

## 6. Experimental Results

### 6.1 Initial Baseline Experiment

Notebook 01 evaluated 100 sentences.

| Model | BLEU | chrF |
|---|---:|---:|
| Dictionary baseline | 0.54 | 15.52 |
| Pretrained Transformer | 57.97 | 71.93 |

The pretrained Transformer substantially outperformed the introductory dictionary baseline.

### 6.2 Final Comparison

All four methods were evaluated on the same **500 validation sentences**.

| Model | BLEU | chrF |
|---|---:|---:|
| Dictionary baseline | 0.66 | 16.14 |
| TF-IDF retrieval | 18.39 | 37.74 |
| LSTM with attention | 28.55 | 50.39 |
| Pretrained Transformer | 55.11 | 70.40 |

### 6.3 Results Interpretation

The dictionary baseline achieved the lowest scores because its vocabulary was very limited and it did not model grammatical relationships.

TF-IDF retrieval improved performance by using similarities between English sentences and retrieving corresponding French translations.

The LSTM achieved higher scores than both simpler baselines. Its results indicate that it learned useful English-to-French translation patterns from the parallel training data.

The pretrained Transformer achieved the highest BLEU and chrF scores.

Compared with the LSTM, the Transformer achieved:

- **26.56 additional BLEU points**
- **20.01 additional chrF points**

However, the Transformer was already pretrained using substantial translation data, while the LSTM was trained from scratch on 50,000 sentence pairs.

Therefore, the observed performance difference cannot be attributed exclusively to the architectures.

### 6.4 Transformer Decoding Comparison

Notebook 02 also compared different Transformer decoding methods.

| Method | BLEU | chrF |
|---|---:|---:|
| Greedy decoding | 55.11 | 70.40 |
| Beam search, 3 beams | 55.19 | 70.39 |
| Beam search, 5 beams | 54.87 | 70.12 |

Beam search provided little improvement in translation quality while increasing the measured inference time.

The main comparison therefore uses the greedy Transformer predictions.

## 7. Qualitative Translation Analysis

The project also examined differences between generated translations and reference sentences.

Three common sources of difficulty are:

- **Vocabulary limitations:** Dictionary-based systems cannot translate many unseen words.
- **Retrieval limitations:** TF-IDF can select a sentence that is lexically similar but has a different intended meaning.
- **Neural translation errors:** LSTM and Transformer outputs may contain incorrect vocabulary choices, omissions, or grammatical problems.

BLEU and chrF may also penalize valid translations that express the same meaning using different words.

Therefore, numerical scores should be complemented by careful inspection of generated translations.

## 8. Project Limitations

### 8.1 Unequal Training Conditions

The Transformer benefited from extensive pretraining, while the LSTM was trained from scratch.

This is a comparison of the implemented translation systems, not a controlled experiment isolating architecture alone.

### 8.2 Training Data Size

The LSTM used 50,000 training examples rather than the full training partition.

Its performance might change with additional data or training.

### 8.3 Evaluation Scope

The main comparison uses 500 validation sentences from the project corpus.

The comparison does not establish performance on all translation domains or longer, more complex texts.

### 8.4 Evaluation Metrics

BLEU and chrF focus on reference overlap and cannot fully assess semantic accuracy, fluency, or acceptability.

### 8.5 Historical Pretraining Data

Although the dataset split reduced overlap between local partitions, overlap with the Transformer's historical pretraining data cannot be completely ruled out.

### 8.6 Computational Comparison

The LSTM inference run used a GPU, while the recorded Transformer decoding timings in Notebook 02 used a CPU.

Consequently, their recorded inference times are not directly comparable.

## 9. Conclusion

This project investigated four English-to-French translation methods: a dictionary baseline, TF-IDF retrieval, an LSTM encoder–decoder with attention, and a pretrained Transformer.

The LSTM achieved **28.55 BLEU** and **50.39 chrF**, outperforming both traditional baselines.

The pretrained Transformer obtained the highest translation-quality scores, achieving **55.11 BLEU** and **70.40 chrF** on the same 500 validation sentences.

These results demonstrate the value of neural sequence-to-sequence translation and show that the pretrained Transformer system achieved the best performance among the tested approaches.

The experiment also illustrates the importance of training data, pretraining, attention mechanisms, careful dataset separation, and consistent evaluation when comparing machine translation systems.
