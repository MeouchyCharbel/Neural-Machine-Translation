
# Neural Machine Translation — English to French

## Project Overview

This project investigates English-to-French Neural Machine Translation (NMT) using a pretrained Transformer model, traditional baselines, fine-tuning, and quantitative evaluation.

The main objectives are to:

1. Compare a simple dictionary baseline with a pretrained Transformer.
2. Develop a stronger TF-IDF retrieval baseline.
3. Prevent exact and normalized data leakage between dataset partitions.
4. Compare greedy decoding and beam search.
5. Fine-tune the pretrained Transformer.
6. Evaluate the impact of fine-tuning using BLEU and chrF.
7. Measure uncertainty using paired bootstrap confidence intervals.
8. Test generalization on an external dataset.

## Models

The main pretrained model is:

`Helsinki-NLP/opus-mt-en-fr`

Three Transformer configurations were evaluated:

- Original pretrained Transformer
- Fine-tuned Transformer after a 100-step pilot
- Extended fine-tuned Transformer

The final selected model is the extended fine-tuned checkpoint.

## Dataset

The main dataset is an English-French parallel corpus from the TensorFlow-hosted `fra-eng.zip` archive.

Dataset preparation includes:

- Removing missing and empty sentence pairs
- Removing duplicate English-French pairs
- Keeping English sentences containing at most 15 words
- Normalizing English and French text
- Grouping related sentences
- Creating group-separated training, validation, and test sets

### Dataset statistics

| Partition | Sentence pairs |
|---|---:|
| Full cleaned dataset | 166,340 |
| Training | 133,017 |
| Validation | 16,616 |
| Test | 16,707 |

The fine-tuning experiments used 20,000 selected training pairs and 1,000 validation pairs.

The dataset separation prevents exact and normalized overlaps according to the implemented grouping rules. It does not guarantee the absence of all semantic near-duplicates or overlap with historical pretrained-model data.

## Notebook Structure

### 01 — Data and Baseline

`01_data_and_baseline.ipynb`

- Loads and cleans the dataset
- Implements a word-by-word dictionary translator
- Loads the pretrained Transformer
- Generates English-to-French translations
- Evaluates BLEU and chrF
- Compares sample translations

### 02 — Evaluation Improvements

`02_evaluation_improvements.ipynb`

- Implements group-aware dataset splitting
- Checks for exact and normalized overlap
- Screens sampled data for near-duplicates
- Implements TF-IDF retrieval translation
- Compares dictionary, retrieval, and Transformer methods
- Evaluates greedy decoding and beam search
- Creates a systematic translation error-review dataset

### 03 — Fine-Tuning

`03_finetuning.ipynb`

- Prepares supervised training examples
- Fine-tunes the pretrained Transformer for 100 optimizer steps
- Continues training from the pilot checkpoint for one additional epoch
- Uses validation loss to monitor training
- Compares original, pilot, and extended models
- Performs paired group bootstrap analysis
- Supports final held-out testing and GPU benchmarking

### 04 — External Evaluation

`04_external_evaluation.ipynb`

- Loads the FLORES-200 English-French devtest benchmark
- Evaluates the original and extended fine-tuned Transformers
- Calculates corpus BLEU and chrF
- Compares translation predictions
- Estimates paired bootstrap confidence intervals

## Evaluation Metrics

### BLEU

BLEU evaluates translation quality through token n-gram overlap with reference translations and includes a brevity penalty.

### chrF

chrF evaluates character n-gram precision and recall against reference translations.

Higher BLEU and chrF scores indicate greater reference similarity, but neither metric directly guarantees semantic correctness.

## Experimental Results

### Original baseline experiment

The first notebook evaluated 100 selected sentences.

| Model | BLEU | chrF |
|---|---:|---:|
| Dictionary baseline | 0.54 | 15.52 |
| Pretrained Transformer | 57.97 | 71.93 |

### Improved validation baseline

The second notebook evaluated 500 validation sentences.

| Method | BLEU | chrF |
|---|---:|---:|
| Dictionary baseline | 0.66 | 16.14 |
| TF-IDF retrieval | 18.39 | 37.74 |
| Pretrained Transformer, greedy | 55.11 | 70.40 |

### Decoding comparison

| Method | BLEU | chrF |
|---|---:|---:|
| Greedy | 55.11 | 70.40 |
| Beam search, 3 beams | 55.19 | 70.39 |
| Beam search, 5 beams | 54.87 | 70.12 |

Greedy decoding was selected because additional beam search provided little quality improvement while increasing inference time.

### Fine-tuning validation results

All three model versions were evaluated on the same 500 validation sentences using greedy decoding.

| Model | BLEU | chrF |
|---|---:|---:|
| Original pretrained | 55.11 | 70.40 |
| Fine-tuned, 100 steps | 55.84 | 71.62 |
| Extended fine-tuned | 56.56 | 72.71 |

### Final held-out test

Evaluation on 2,000 held-out sentence pairs:

| Model | BLEU | chrF |
|---|---:|---:|
| Original pretrained | 52.22 | 68.03 |
| Extended fine-tuned | 55.74 | 71.58 |

The extended model improved BLEU by 3.52 points and chrF by 3.55 points on this test sample.

### External FLORES-200 evaluation

Evaluation on 1,012 English-French sentences from FLORES-200 devtest:

| Model | BLEU | chrF |
|---|---:|---:|
| Original pretrained | 46.51 | 69.00 |
| Extended fine-tuned | 46.86 | 69.23 |

Observed improvements were approximately +0.35 BLEU and +0.23 chrF.

Paired bootstrap analysis with 1,000 repetitions produced:

| Metric | Improvement | 95% confidence interval |
|---|---:|---|
| BLEU | +0.351 | [-0.222, 0.960] |
| chrF | +0.236 | [-0.092, 0.620] |

Both intervals include zero. The external improvements are therefore statistically inconclusive under this bootstrap procedure.

## Inference Performance

A controlled GPU benchmark used 500 validation sentences, batch size 8, and three timed trials per model.

| Model | Median GPU time | Sentences/second |
|---|---:|---:|
| Original pretrained | 8.72 s | 57.37 |
| Fine-tuned, 100 steps | 8.78 s | 56.94 |
| Extended fine-tuned | 9.08 s | 55.06 |

Small inference-time differences should not be interpreted as a consistent hardware-speed advantage without additional repeated testing.

## Technologies

- Python
- Google Colab
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- SentencePiece
- Pandas and NumPy
- scikit-learn
- SacreBLEU
- Matplotlib
- Google Drive for model checkpoints
- GitHub for version control and results

## How to Run

1. Open the required notebook in Google Colab.
2. Install the dependencies using its first setup cells.
3. Enable a T4 GPU for fine-tuning and GPU-based inference.
4. Run the notebook cells in order.
5. For Notebook 03 and Notebook 04, mount Google Drive when prompted.
6. Ensure the saved fine-tuned checkpoints are available in the expected Google Drive paths.

The checkpoints were stored under:

`/content/drive/MyDrive/NMT_Finetuning/`

The checkpoints are not included as model-weight files in this GitHub repository.

The evaluation results are available in `results/`.

## Limitations

- The main corpus focuses on short English sentences.
- The original pretrained model may have encountered some evaluation material during historical pretraining.
- BLEU and chrF do not fully capture meaning, fluency, or factual correctness.
- Qualitative translation judgments require careful human verification.
- The external evaluation showed only small, statistically inconclusive improvements.
- Results should not be generalized automatically to all domains or sentence lengths.
- Trained checkpoints are stored separately from the repository.

## Conclusion

The pretrained Transformer substantially outperformed the dictionary and TF-IDF retrieval baselines.

Fine-tuning improved BLEU and chrF on the original corpus's held-out test sample.

On the external FLORES-200 benchmark, the fine-tuned model achieved slightly higher scores, but paired bootstrap confidence intervals included zero.

These experiments demonstrate the value of pretrained Transformers, systematic evaluation, fine-tuning, and independent cross-dataset testing for neural machine translation.
