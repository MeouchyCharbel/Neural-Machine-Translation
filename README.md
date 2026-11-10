
# Neural Machine Translation — English to French

## Project Overview

This project investigates English-to-French neural machine translation (NMT). It develops simple dictionary and TF-IDF retrieval baselines, examines decoding with a pretrained MarianMT Transformer, and trains two LSTM encoder–decoder variants **with and without attention**.

The **main final experiment** compares three neural translation systems on the **same 2,000 held-out test sentences**:

1. LSTM without attention — trained from scratch
2. LSTM with attention — trained from scratch
3. Pretrained MarianMT Transformer — used without project-specific fine-tuning

Translation quality is measured with corpus BLEU and chrF. Earlier notebook results on 100 or 500 sentences are retained as preliminary experiments and should not be directly compared with the final 2,000-sentence test scores.

## Project Objectives

- Clean and split an English–French parallel corpus while limiting data leakage.
- Implement dictionary-based and TF-IDF retrieval baselines.
- Evaluate a pretrained MarianMT Transformer and its decoding strategies.
- Implement LSTM encoder–decoder systems with and without cross-attention.
- Train the two LSTMs using matched training data and hyperparameters.
- Compare all three neural models on identical held-out test examples.
- Evaluate translations with BLEU, chrF, and qualitative error inspection.
- Discuss the effects of attention, model training conditions, and the limits of the comparisons.

## Repository Structure

```text
Neural-Machine-Translation/
├── 01_data_and_baseline.ipynb
├── 02_evaluation_improvements.ipynb
├── 03_lstm_three_model_comparison.ipynb
├── fra.txt
├── requirements.txt
├── README.md
├── report/
│   └── NMT_Report.md
└── results/
    ├── metrics.csv
    ├── baseline_metrics.csv
    ├── decoding_metrics.csv
    ├── validation_predictions.csv
    ├── three_model_test_metrics.csv
    ├── three_model_test_predictions.csv
    ├── lstm_training_history.csv
    └── three_model_test_metadata.json
```

Other error-review and translation-comparison files may also be present in `results/`.

### Notebook 01 — Data and Baseline

**File:** `01_data_and_baseline.ipynb`

- Loads and prepares English–French examples.
- Implements an introductory dictionary translator.
- Loads `Helsinki-NLP/opus-mt-en-fr` (MarianMT).
- Compares dictionary and Transformer predictions using BLEU and chrF on an initial 100-sentence sample.

### Notebook 02 — Evaluation Improvements

**File:** `02_evaluation_improvements.ipynb`

- Creates group-aware training, validation, and test partitions and checks for overlapping examples.
- Adds TF-IDF retrieval using cosine similarity; the index contains **92,892 unique English training sentences**.
- Compares dictionary, TF-IDF, and MarianMT on **500 validation sentences**.
- Studies Transformer greedy decoding versus beam search with 3 or 5 beams.
- Saves predictions and supports qualitative error analysis.

### Notebook 03 — Three-Model Neural Comparison

**File:** `03_lstm_three_model_comparison.ipynb`

- Reconstructs the dataset partition and verifies group-level train/validation/test separation.
- Selects **50,000 training sentence pairs**, **1,000 validation-monitor examples**, and **2,000 held-out test sentences**.
- Trains a bidirectional-encoder LSTM **without attention**.
- Trains a matched bidirectional-encoder LSTM **with four-head cross-attention**.
- Selects the best checkpoint of each LSTM using monitoring loss.
- Runs pretrained MarianMT on the exact same 2,000 test inputs.
- Uses greedy decoding for each model and reports corpus BLEU and chrF.
- Exports per-sentence translations, metrics, training histories, and experiment metadata.
- Saves model checkpoints and SentencePiece tokenizers to Google Drive for reuse.

## Dataset and Preprocessing

**Source:** TensorFlow-hosted English–French `fra-eng.zip` parallel sentence corpus.

Dataset preparation removes empty rows, duplicate pairs, and sentences outside the project's length constraint, and performs normalized overlap detection. Related English/French examples are grouped to reduce leakage across partitions.

| Stage / partition | Sentence pairs |
|---|---:|
| Original corpus | 167,130 |
| Cleaned corpus | 166,340 |
| Training partition | 133,017 |
| Validation partition | 16,616 |
| Test partition | 16,707 |

**Final Notebook 03 selection:** 50,000 pairs from training, 1,000 examples from validation for checkpoint selection, and 2,000 randomly sampled examples from the held-out test partition. The test set is not used for model training or checkpoint selection. The notebook checks that the relevant split groups do not overlap.

## Translation Approaches

### Dictionary Baseline

A small hand-built English–French word mapping translates tokens individually. It is a useful minimal baseline but has limited vocabulary, grammatical coverage, and contextual awareness.

### TF-IDF Retrieval

The English source sentence is represented by a TF-IDF vector and compared with training sentences by cosine similarity. The French translation associated with the closest training match is retrieved; the method does not generate a new sentence token by token.

### LSTM without Attention

The model uses trainable English and French subword embeddings, a bidirectional LSTM encoder, and an autoregressive LSTM decoder. The decoder is initialized using the encoder's final hidden and cell states. It does not consult the full sequence of encoder outputs through an attention layer.

### LSTM with Attention

The model shares the general encoder–decoder design, vocabulary, training examples, and training configuration of the non-attention LSTM. It additionally applies **four-head cross-attention** from decoder representations to encoder outputs when predicting French tokens.

### Pretrained Transformer (MarianMT)

The pretrained English-to-French model is `Helsinki-NLP/opus-mt-en-fr`. It is used for inference **without additional fine-tuning** in this project. Its historical pretraining data and computational budget differ from those of the LSTMs trained from scratch.

## Training Configuration — Final LSTM Experiment

| Setting | Value |
|---|---|
| Training pairs per LSTM | 50,000 |
| Monitoring examples | 1,000 |
| Epochs | Up to 5 |
| Batch size | 64 |
| Embedding / hidden dimensions | 256 / 256 |
| Tokenization | Separate SentencePiece BPE vocabularies |
| Vocabulary size | 6,000 per language |
| Maximum LSTM sequence length | 48 tokens |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Objective | Cross-entropy (padding tokens ignored) |
| Training method | Teacher forcing, gradient clipping |
| Checkpoint selection | Lowest validation-monitor loss |
| Early-stopping patience | 2 epochs |
| Seed | 42 |

The two LSTMs have matched training settings, although adding attention also changes the number of trainable parameters.

| Model | Best epoch | Best monitoring loss |
|---|---:|---:|
| LSTM without attention | 5 | 2.4552 |
| LSTM with attention | 4 | 1.7427 |

Both LSTMs were trained for five epochs in the recorded run, then their best checkpoints were used for inference.

## Evaluation Metrics

- **BLEU** measures overlap between generated and reference translations using word n-grams and a brevity penalty.
- **chrF** measures character n-gram precision and recall against references.

Higher scores generally indicate closer reference agreement. Neither score is an accuracy percentage or a direct guarantee of semantic correctness; valid paraphrases may receive lower scores.

## Experimental Results

### Main Final Result — 2,000 Held-Out Test Sentences

**Source:** `results/three_model_test_metrics.csv`.

| Model | BLEU | chrF |
|---|---:|---:|
| LSTM without attention | 10.46 | 28.24 |
| LSTM with attention | 31.04 | 51.10 |
| Pretrained MarianMT Transformer | **53.74** | **68.69** |

All three models translated the same 2,000 English inputs and were scored against the same French reference translations with greedy decoding.

**Observed differences:**

- Adding attention to the LSTM coincided with a **20.58-point BLEU** and **22.86-point chrF** increase over the non-attention LSTM.
- MarianMT scored **22.70 BLEU points** and **17.59 chrF points** above the attention-based LSTM.
- These differences are descriptive results from the recorded run, **not** formal statistical-significance claims.

The LSTM-to-LSTM comparison is more controlled than the LSTM-to-MarianMT comparison because the LSTMs share the training corpus and main training settings. However, they differ in parameterization, and MarianMT benefits from external pretraining. Therefore, this experiment does not isolate architecture as the only cause of score differences.

### Earlier Experiment — 500 Validation Sentences (Not the Final Test)

**Sources:** `results/lstm_comparison_metrics.csv` and `results/validation_predictions.csv`.

| Model | BLEU | chrF |
|---|---:|---:|
| Dictionary baseline | 0.66 | 16.14 |
| TF-IDF retrieval | 18.39 | 37.74 |
| LSTM with attention (previous run) | 28.55 | 50.39 |
| Pretrained MarianMT | 55.11 | 70.40 |

This earlier four-method comparison used **validation** sentences rather than the final test sample. Its scores should **not** be combined with or presented as equivalent to the 2,000-sentence test results.

### Earlier Experiment — Transformer Decoding (500 Validation Sentences)

| MarianMT decoding | BLEU | chrF |
|---|---:|---:|
| Greedy | 55.11 | 70.40 |
| Beam search (3 beams) | 55.19 | 70.39 |
| Beam search (5 beams) | 54.87 | 70.12 |

Beam search provided little improvement on this validation sample, so the final three-model test uses greedy decoding for all models.

### Initial Experiment — Notebook 01 (100 Sentences)

| Model | BLEU | chrF |
|---|---:|---:|
| Dictionary baseline | 0.54 | 15.52 |
| MarianMT | 57.97 | 71.93 |

These initial results are exploratory and use a different evaluation sample.

## Results Files

The `results/` folder stores outputs from different stages:

| File | Contents |
|---|---|
| `three_model_test_metrics.csv` | **Final 2,000-test-sentence BLEU/chrF comparison** |
| `three_model_test_predictions.csv` | English sources, French references, and the three model outputs |
| `lstm_training_history.csv` | Training and validation-monitor losses by epoch for both LSTMs |
| `three_model_test_metadata.json` | Data counts, model configurations, test selection, and decoding settings |
| `validation_predictions.csv` | Earlier 500-sentence validation predictions |
| `lstm_comparison_metrics.csv` | Earlier four-method validation metrics |
| `lstm_comparison_predictions.csv` | Earlier four-method validation translations |
| `baseline_metrics.csv` | Dictionary and TF-IDF baseline metrics |
| `decoding_metrics.csv` | MarianMT decoding-strategy evaluation |
| `metrics.csv` | Initial notebook metrics |

Model checkpoints and SentencePiece `.model` files are saved by Notebook 03 into the Google Drive folder `MyDrive/NMT_Three_Model_Comparison`. These large checkpoint files are not necessary to store in the Git repository.

## How to Run

1. Open the notebooks in Google Colab (a GPU runtime is recommended for Notebook 03).
2. Install dependencies using `requirements.txt` or the setup cell in each notebook.
3. Run `01_data_and_baseline.ipynb` to explore the original baseline.
4. Run `02_evaluation_improvements.ipynb` to reproduce TF-IDF and Transformer decoding experiments.
5. Run `03_lstm_three_model_comparison.ipynb` from the first cell for the final neural-model experiment.
6. Authorize Google Drive mounting if prompted; the final notebook saves or restores its model checkpoints there.
7. Inspect `results/three_model_test_metrics.csv`, `results/three_model_test_predictions.csv`, and `results/lstm_training_history.csv`.

The final notebook downloads the parallel corpus itself; `fra.txt` is also retained in the repository as a dataset resource.

## Tools and Libraries

Python, Jupyter/Google Colab, PyTorch, Hugging Face Transformers, SentencePiece, SacreBLEU, scikit-learn, pandas, NumPy, Matplotlib, and GitHub.

## Limitations

1. **Unequal pretraining:** MarianMT was pretrained elsewhere; both LSTMs were initialized and trained using this project's data. Higher MarianMT scores cannot be credited to architecture alone.
2. **Different model capacities:** The with-attention LSTM has additional trainable layers/parameters, so the LSTM comparison is not a strictly equal-parameter ablation.
3. **Training size:** The two LSTMs used 50,000 pairs, not the full cleaned training partition.
4. **Test scope:** The final scores describe a 2,000-sentence sample from this corpus; they do not establish universal performance across domains.
5. **Metric limitations:** BLEU and chrF measure reference agreement, not perfect semantic correctness or fluency.
6. **Pretraining overlap:** Overlap with MarianMT's external pretraining data cannot be definitively excluded.
7. **Runtime results:** Inference timing depends on hardware, batch size, tokenization, implementation, and model generation settings; speed figures should not be overinterpreted.
8. **Uncertainty:** No formal statistical-significance or confidence-interval analysis is included for the new three-model test comparison.

## Conclusion

On a shared set of **2,000 held-out test sentences**, the LSTM without attention scored **10.46 BLEU / 28.24 chrF**, the LSTM with attention scored **31.04 BLEU / 51.10 chrF**, and pretrained MarianMT scored **53.74 BLEU / 68.69 chrF**.

In this implementation, adding cross-attention substantially improved the LSTM's reference-based translation quality. Pretrained MarianMT performed best overall, with the important qualification that its pretraining conditions differ from the LSTMs. The project demonstrates the value of consistent evaluation and careful interpretation of model comparisons in neural machine translation.
