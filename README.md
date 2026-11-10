# Neural Machine Translation — English to French

## Project Overview

This project investigates English-to-French Neural Machine Translation (NMT) by comparing a pretrained Transformer with an LSTM encoder–decoder model and two simpler translation baselines.

The main objective is to evaluate the performance of different translation approaches using BLEU and chrF scores.

The project compares four methods:

1. Dictionary-based translation
2. TF-IDF retrieval-based translation
3. LSTM encoder–decoder with attention
4. Pretrained Transformer (MarianMT)

All four methods are evaluated using the same 500 validation sentences to ensure a consistent comparison.

## Project Objectives

- Prepare and clean an English–French parallel dataset.
- Implement a simple dictionary translation baseline.
- Implement a TF-IDF retrieval-based baseline.
- Evaluate an existing pretrained Transformer.
- Compare greedy decoding and beam search.
- Build and train an LSTM encoder–decoder model with an attention mechanism.
- Evaluate translation quality using BLEU and chrF.
- Compare model predictions and analyze translation errors.
- Identify the advantages and limitations of each translation approach.

## Project Structure

The project consists of three Jupyter notebooks.

### Notebook 01 — Data and Baseline

**File:** `01_data_and_baseline.ipynb`

This notebook:

- Loads the English–French dataset.
- Cleans and preprocesses the sentence pairs.
- Implements a simple dictionary-based translator.
- Loads the pretrained Transformer.
- Generates French translations.
- Evaluates translation quality using BLEU and chrF.
- Compares the dictionary baseline with the Transformer.

### Notebook 02 — Evaluation Improvements

**File:** `02_evaluation_improvements.ipynb`

This notebook:

- Implements group-aware dataset splitting to reduce data leakage.
- Checks for overlapping sentences between dataset partitions.
- Implements a TF-IDF retrieval baseline.
- Compares dictionary, TF-IDF, and Transformer translation methods.
- Evaluates greedy decoding and beam search.
- Calculates BLEU, chrF, and inference time.
- Analyzes translation errors.

### Notebook 03 — LSTM Baseline

**File:** `03_lstm_baseline.ipynb`

This notebook:

- Prepares English–French sentence pairs for neural training.
- Builds a bidirectional LSTM encoder.
- Builds an LSTM decoder with an attention mechanism.
- Trains the LSTM using supervised learning.
- Monitors training and validation loss.
- Generates French translations using the trained model.
- Evaluates the LSTM on the same 500 validation sentences as Notebook 02.
- Compares the LSTM with the dictionary, TF-IDF, and pretrained Transformer.
- Saves evaluation metrics and translation predictions.

## Dataset

The project uses an English–French parallel corpus from the TensorFlow-hosted `fra-eng.zip` dataset.

Each example contains an English sentence and its corresponding French translation.

### Data Preprocessing

The preprocessing steps include:

- Removing missing and empty sentence pairs.
- Removing duplicate translation pairs.
- Removing examples with English sentences longer than 15 words.
- Normalizing text for overlap detection.
- Grouping related sentence pairs to reduce data leakage.
- Splitting the data into training, validation, and test partitions.

### Dataset Statistics

| Dataset Partition | Sentence Pairs |
|---|---:|
| Cleaned dataset | 166,340 |
| Training | 133,017 |
| Validation | 16,616 |
| Test | 16,707 |

For the LSTM experiment, 50,000 training pairs were used.

The LSTM and Transformer were compared on the same 500 validation examples.

## Translation Models

### 1. Dictionary Baseline

The dictionary baseline translates English sentences word by word using predefined English–French word mappings.

It is computationally inexpensive but has important limitations:

- Limited vocabulary
- No contextual understanding
- No grammatical modeling
- Difficulty translating unknown words

This baseline provides a simple reference point.

### 2. TF-IDF Retrieval Baseline

The TF-IDF baseline represents English sentences as numerical vectors.

For each input sentence, it searches for the most similar sentence in the training dataset using cosine similarity.

It returns the French translation associated with the closest English sentence.

The retrieval system indexed 92,892 unique English training sentences.

Unlike neural models, this approach retrieves existing translations instead of generating new ones.

### 3. LSTM Encoder–Decoder with Attention

An LSTM (Long Short-Term Memory) is a recurrent neural network designed to process sequential information.

The translation model uses:

- A bidirectional LSTM encoder to process English sentences.
- An LSTM decoder to generate French sentences.
- An attention mechanism to help the decoder focus on relevant encoder outputs.
- Embedding layers to represent tokens as numerical vectors.

The LSTM was trained from scratch using English–French sentence pairs.

#### LSTM Training Configuration

| Parameter | Value |
|---|---|
| Training examples | 50,000 |
| Training epochs | 5 |
| Best epoch | 4 |
| Embedding dimension | 256 |
| Batch size | 64 |
| Vocabulary size | 6,000 subword entries per language |
| Optimizer | AdamW |
| Best validation-monitor loss | 1.7374 |
| Trainable parameters | 7,466,352 |

The training procedure includes teacher forcing, cross-entropy loss, gradient clipping, validation monitoring, and early stopping.

### 4. Pretrained Transformer

The Transformer model used in this project is:

`Helsinki-NLP/opus-mt-en-fr`

This is a pretrained MarianMT encoder–decoder Transformer designed for English-to-French translation.

Unlike recurrent models, Transformers use attention mechanisms to model relationships between tokens without processing the entire sequence through recurrent hidden-state updates.

The model was used without additional fine-tuning in the revised project.

## Evaluation Metrics

### BLEU — Bilingual Evaluation Understudy

BLEU evaluates translation quality by comparing word n-grams in generated translations with reference translations.

Higher BLEU scores indicate greater agreement with reference translations.

BLEU also includes a brevity penalty for translations that are too short.

### chrF — Character n-gram F-score

chrF evaluates translation quality using character n-gram precision and recall.

It is useful for identifying similarities between words and their morphological forms.

Both BLEU and chrF evaluate reference overlap rather than directly measuring semantic correctness.

## Experimental Results

### Notebook 01 — Initial Baseline Comparison

The first notebook evaluates 100 selected sentences.

| Model | BLEU | chrF |
|---|---:|---:|
| Dictionary Baseline | 0.54 | 15.52 |
| Pretrained Transformer | 57.97 | 71.93 |

The Transformer substantially outperformed the small dictionary baseline.

### Final Model Comparison — 500 Validation Sentences

| Model | BLEU | chrF |
|---|---:|---:|
| Dictionary Baseline | 0.66 | 16.14 |
| TF-IDF Retrieval | 18.39 | 37.74 |
| LSTM with Attention | 28.55 | 50.39 |
| Pretrained Transformer | 55.11 | 70.40 |

### Results Analysis

The dictionary baseline achieved the lowest translation-quality scores because of its very limited vocabulary and lack of contextual understanding.

The TF-IDF retrieval baseline improved translation quality by retrieving similar sentence pairs from the training dataset.

The LSTM with attention achieved higher BLEU and chrF scores than both traditional baselines, demonstrating its ability to learn translation patterns from parallel sentence data.

The pretrained Transformer achieved the highest scores, outperforming the LSTM on the selected validation set.

However, the Transformer was pretrained on a much larger amount of data, whereas the LSTM was trained from scratch on 50,000 examples.

Therefore, this experiment does not isolate the effect of architecture alone.

### Transformer Decoding Comparison

| Decoding Method | BLEU | chrF |
|---|---:|---:|
| Greedy | 55.11 | 70.40 |
| Beam Search 3 | 55.19 | 70.39 |
| Beam Search 5 | 54.87 | 70.12 |

Beam search did not produce a substantial improvement over greedy decoding on the selected validation sentences.

Greedy decoding was therefore used for the main Transformer comparison.

## Technologies Used

- Python
- Jupyter Notebook
- Google Colab
- PyTorch
- Hugging Face Transformers
- SentencePiece
- Scikit-learn
- Pandas
- NumPy
- SacreBLEU
- Matplotlib
- GitHub

## How to Run the Project

1. Open the notebooks in Google Colab or another compatible Jupyter environment.
2. Install the required Python dependencies.
3. Run Notebook 01 to reproduce the initial dataset and baseline experiment.
4. Run Notebook 02 to reproduce the improved baselines and Transformer evaluation.
5. Open Notebook 03 and enable a GPU runtime when available.
6. Run the LSTM training and evaluation cells in order.
7. Review the BLEU and chrF comparison results.
8. Examine the generated translations and error examples.

Notebook 03 reproduces the dataset split and checks that its 500 evaluation examples match the saved predictions from Notebook 02.

## Results Files

The `results/` directory contains evaluation outputs.

Important files include:

- `metrics.csv` — Initial dictionary and Transformer results.
- `baseline_metrics.csv` — Dictionary and TF-IDF evaluation metrics.
- `decoding_metrics.csv` — Transformer greedy and beam-search results.
- `validation_predictions.csv` — Predictions on the shared validation sample.
- `lstm_comparison_metrics.csv` — Final comparison metrics, including LSTM results.
- `lstm_comparison_predictions.csv` — Translation examples and predictions from the compared methods.

Older fine-tuning and external-evaluation CSVs, if retained in the repository, are historical artifacts and are not part of the revised experiment.

## Project Limitations

### Different Training Conditions

The pretrained Transformer and the LSTM do not have equivalent training conditions.

The Transformer benefits from extensive pretraining, while the LSTM was trained from random initialization on a smaller dataset.

Consequently, the difference in performance cannot be attributed entirely to the network architecture.

### Evaluation Dataset

The final model comparison uses 500 validation sentences from the main dataset.

These results do not necessarily represent performance on all real-world English–French translation tasks.

### Evaluation Metrics

BLEU and chrF do not fully capture translation meaning or fluency.

Correct paraphrases may obtain lower scores because their wording differs from the reference translation.

### Dictionary Limitations

The original dictionary contains only 11 predefined translations.

It is an introductory baseline rather than a competitive translation method.

## Conclusion

This project compared four English-to-French translation approaches: dictionary-based translation, TF-IDF retrieval, an attention-based LSTM, and a pretrained Transformer.

The results demonstrated a clear improvement from the dictionary baseline to TF-IDF retrieval and then to the LSTM.

The LSTM achieved **28.55 BLEU** and **50.39 chrF**, outperforming both simpler baselines.

The pretrained Transformer achieved the highest translation quality, with **55.11 BLEU** and **70.40 chrF** on the same 500 validation sentences.

These findings demonstrate the advantages of neural sequence-to-sequence translation compared with simpler word-based and retrieval-based methods.

The comparison also highlights the importance of pretraining, dataset size, model architecture, and consistent evaluation procedures in Neural Machine Translation.
