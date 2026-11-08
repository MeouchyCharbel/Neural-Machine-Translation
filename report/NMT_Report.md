# Neural Machine Translation: English-to-French Translation Using Pretrained and Fine-Tuned Transformers

## 1. Introduction

Neural Machine Translation (NMT) uses artificial neural networks to translate text between languages. Unlike traditional word-by-word translation, NMT models generate target-language sequences while considering linguistic context and relationships between words.

This project investigates English-to-French translation using the pretrained `Helsinki-NLP/opus-mt-en-fr` Transformer.

The study evaluates traditional translation baselines, compares decoding strategies, fine-tunes the pretrained Transformer, and examines whether the resulting improvements generalize to an external dataset.

### 1.1 Objectives

The objectives are to:

1. Implement dictionary-based and TF-IDF retrieval translation baselines.
2. Compare these baselines with a pretrained Transformer using BLEU and chrF.
3. Reduce data leakage through group-aware dataset partitioning.
4. Compare greedy decoding with beam search.
5. Fine-tune the pretrained Transformer and evaluate translation quality.
6. Estimate uncertainty using paired bootstrap confidence intervals.
7. Evaluate generalization using an independent external benchmark.

## 2. Dataset and Preprocessing

### 2.1 Main Dataset

The project uses an English–French parallel sentence corpus distributed through the TensorFlow-hosted `fra-eng.zip` archive.

The dataset initially contained **167,130 sentence pairs**.

Preprocessing consisted of removing missing values, empty sentences, duplicate English–French pairs, and examples containing more than 15 English words.

After cleaning, **166,340 sentence pairs** remained.

### 2.2 Dataset Partitioning

A group-aware splitting procedure was implemented to reduce information leakage between partitions.

Both English and French sentences were normalized using Unicode normalization, case folding, whitespace normalization, and punctuation-insensitive comparison.

Related sentence pairs were connected using the Union-Find algorithm. `GroupShuffleSplit` was then used to assign entire connected groups to the same partition.

| Partition | Sentence pairs | Percentage |
|---|---:|---:|
| Training | 133,017 | 79.97% |
| Validation | 16,616 | 9.99% |
| Test | 16,707 | 10.04% |

Exact and normalized overlap checks returned zero overlapping keys between the three partitions.

A separate sampled near-duplicate screen used character-level TF-IDF and cosine similarity.

Although these procedures reduce leakage, they cannot guarantee the absence of semantic paraphrases across partitions or overlap with the pretrained model's historical training material.

## 3. Translation Methods

### 3.1 Dictionary Baseline

A small dictionary-based translator was implemented as the simplest reference method.

English words were converted to lowercase and replaced by predefined French equivalents. Words absent from the dictionary remained untranslated.

This baseline requires little computation but cannot reliably account for grammar, context, or vocabulary outside its predefined mappings.

### 3.2 TF-IDF Retrieval Baseline

A second baseline used TF-IDF representations of English sentences.

The system compared each input against indexed English training sentences and retrieved the French translation associated with the closest match.

Word unigrams and bigrams were used as features.

The retrieval system indexed **92,892 unique English training sentences**.

Unlike neural translation, this method retrieves previously stored translations rather than generating new sentences.

### 3.3 Pretrained Transformer

The neural model was `Helsinki-NLP/opus-mt-en-fr`, an encoder–decoder sequence-to-sequence Transformer.

The model contains approximately **75.1 million parameters**.

English inputs were tokenized and passed through the model to generate French token sequences.

The original pretrained model was used without additional parameter updates during the initial evaluation experiments.

## 4. Evaluation Metrics

Two automatic translation metrics were calculated using SacreBLEU.

**BLEU (Bilingual Evaluation Understudy)** measures token n-gram overlap between generated and reference translations, including a brevity penalty.

**chrF** measures character n-gram precision and recall against reference translations.

Higher values generally indicate greater similarity to reference translations.

Both metrics are limited because different expressions may preserve the same meaning while receiving different scores.

Therefore, quantitative evaluation was supplemented with qualitative translation review.

## 5. Baseline Experiments

### 5.1 Initial Baseline

The initial experiment compared the dictionary baseline with the pretrained Transformer on 100 selected sentences.

| Model | BLEU | chrF |
|---|---:|---:|
| Dictionary baseline | 0.54 | 15.52 |
| Pretrained Transformer | 57.97 | 71.93 |

The pretrained Transformer substantially outperformed the simple dictionary baseline.

### 5.2 Improved Validation Comparison

A subsequent experiment evaluated 500 examples from the group-separated validation partition.

| Method | BLEU | chrF |
|---|---:|---:|
| Dictionary baseline | 0.66 | 16.14 |
| TF-IDF retrieval | 18.39 | 37.74 |
| Pretrained Transformer, greedy | 55.11 | 70.40 |

The TF-IDF retrieval method improved substantially over the small dictionary but remained below the pretrained Transformer.

These results demonstrate the advantage of contextual neural generation over the selected word-replacement and retrieval baselines.

## 6. Decoding Strategy Comparison

Three decoding configurations were evaluated on the same 500 validation examples.

| Decoding method | BLEU | chrF | CPU time |
|---|---:|---:|---:|
| Greedy | 55.11 | 70.40 | 76.34 s |
| Beam search, 3 beams | 55.19 | 70.39 | 123.32 s |
| Beam search, 5 beams | 54.87 | 70.12 | 214.10 s |

Beam search with three beams achieved a very small BLEU improvement, while requiring substantially more inference time.

Increasing the beam count to five did not improve either evaluation metric.

**Greedy decoding was therefore selected** for subsequent model comparisons because it offered the most favorable quality–efficiency trade-off in these experiments.

## 7. Transformer Fine-Tuning

### 7.1 Training Data and Configuration

Fine-tuning used 20,000 sentence pairs sampled from the training partition and 1,000 validation pairs.

The original pretrained model was fine-tuned using Hugging Face Transformers and PyTorch on an NVIDIA Tesla T4 GPU.

The training procedure used supervised English-to-French sequence-to-sequence learning.

All model parameters remained trainable.

### 7.2 Pilot Fine-Tuning

A preliminary training experiment was conducted to verify that parameter adaptation improved validation performance.

| Hyperparameter | Value |
|---|---|
| Optimizer steps | 100 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 16 |
| Learning rate | 2 × 10⁻⁵ |
| Optimizer | AdamW |
| Precision | Mixed FP16 |

The pilot achieved a best validation loss of **0.555836**.

Its selected checkpoint was saved as `checkpoint-100`.

### 7.3 Extended Fine-Tuning

The extended experiment initialized the model from the pilot checkpoint and trained for one additional epoch using the same selected training dataset.

The learning rate was reduced to **1 × 10⁻⁵**, and early-stopping monitoring was enabled.

The experiment completed **1,250 additional optimizer steps**.

Best validation loss improved from **0.555836 to 0.510601**.

The selected extended checkpoint was saved as `checkpoint-1250`.

The extended training initialized a new optimizer rather than restoring the pilot optimizer state.

## 8. Fine-Tuning Validation Results

The original, pilot, and extended models were evaluated on the same 500 validation sentences using greedy decoding.

| Model | BLEU | chrF |
|---|---:|---:|
| Original pretrained | 55.11 | 70.40 |
| Fine-tuned, 100 steps | 55.84 | 71.62 |
| Extended fine-tuned | **56.56** | **72.71** |

The extended fine-tuned model achieved the highest BLEU and chrF scores.

Compared with the original model, the observed improvements were approximately **+1.45 BLEU** and **+2.31 chrF**.

### 8.1 Statistical Uncertainty

A paired group bootstrap analysis was performed using 1,000 resampling repetitions.

| Comparison | Metric | Improvement | 95% confidence interval |
|---|---|---:|---|
| Extended vs original | BLEU | +1.450 | [-0.413, 3.188] |
| Extended vs original | chrF | +2.312 | [1.054, 3.605] |
| Extended vs pilot | BLEU | +0.725 | [-0.819, 2.400] |
| Extended vs pilot | chrF | +1.094 | [0.138, 2.121] |

The chrF confidence intervals excluded zero, while the BLEU intervals included zero.

Thus, the validation bootstrap analysis provided stronger evidence of improvement in chrF than in BLEU.

These findings were used for exploratory model comparison and should not be interpreted as independent final-test evidence.

## 9. Final Held-Out Test Evaluation

After model selection, the original pretrained model and the extended fine-tuned model were evaluated on 2,000 examples from the held-out test partition.

| Model | BLEU | chrF |
|---|---:|---:|
| Original pretrained | 52.22 | 68.03 |
| Extended fine-tuned | **55.74** | **71.58** |

The extended model improved BLEU by **3.52 points** and chrF by **3.55 points**.

These results indicate improved reference-based translation quality on the main corpus's held-out test sample.

The test examples were separate from the project's training and validation partitions according to the implemented grouping rules.

However, historical overlap with the pretrained model's external training data cannot be ruled out.

## 10. External Evaluation Using FLORES-200

### 10.1 Motivation

Fine-tuning improvements on one corpus may not generalize equally well to sentences from other sources.

Therefore, an additional experiment evaluated both models on the English–French FLORES-200 `devtest` benchmark.

The external evaluation contained **1,012 parallel sentence pairs**.

No additional training was performed on FLORES-200.

### 10.2 Results

| Model | BLEU | chrF |
|---|---:|---:|
| Original pretrained | 46.51 | 69.00 |
| Extended fine-tuned | **46.86** | **69.23** |

The fine-tuned model achieved small observed gains of approximately +0.35 BLEU and +0.23 chrF.

### 10.3 Bootstrap Analysis

A paired bootstrap analysis with 1,000 resampling repetitions was conducted to estimate the uncertainty of the external performance differences.

| Metric | Observed improvement | 95% confidence interval |
|---|---:|---|
| BLEU | +0.351 | [-0.222, 0.960] |
| chrF | +0.236 | [-0.092, 0.620] |

Both confidence intervals included zero.

Consequently, the external experiment did not provide sufficiently strong statistical evidence that fine-tuning improved either metric on FLORES-200.

The fine-tuned model nevertheless produced comparable results, with slightly higher observed scores.

### 10.4 Interpretation

The results suggest that the benefits of fine-tuning were more pronounced on the original corpus's held-out test sample than on the external benchmark.

Differences in sentence structure, length, vocabulary, and domain may contribute to this result.

The experiment illustrates the importance of evaluating machine-translation systems on datasets beyond those used during model development.

## 11. Inference Performance

A controlled GPU inference benchmark compared the three model configurations using 500 sentences, a batch size of eight, and three timing trials per model.

| Model | Median time | Sentences per second |
|---|---:|---:|
| Original pretrained | 8.72 s | 57.37 |
| Fine-tuned, 100 steps | 8.78 s | 56.94 |
| Extended fine-tuned | 9.08 s | 55.06 |

All configurations demonstrated broadly comparable throughput.

The extended model's slightly lower throughput does not imply a change in architecture or parameter count.

Small timing differences can reflect execution conditions and differences in generated output lengths.

## 12. Qualitative Error Analysis

Automatic evaluation metrics cannot fully determine whether a translation preserves the intended meaning.

Therefore, the project included systematic translation reviews.

One review examined examples with low chrF, decoding disagreements, and randomly selected sentences.

A later AI-assisted review compared the original, pilot, and extended models on 40 deliberately selected examples.

That review classified four cases as improved, three as worsened, and 33 as semantically equivalent.

These judgments were preliminary and were not independently verified by human annotators.

Because the examples were selected using a stratified strategy emphasizing translation differences, these counts cannot be interpreted as population-level improvement rates.

## 13. Discussion

The results support several findings.

First, the pretrained Transformer substantially outperformed both traditional baselines, indicating the importance of contextual sequence generation for English-to-French translation.

Second, greedy decoding provided a reasonable balance of translation quality and efficiency compared with the tested beam-search configurations.

Third, fine-tuning improved validation scores and the final held-out test metrics on the original parallel corpus.

Finally, external evaluation on FLORES-200 showed only small positive differences, with bootstrap confidence intervals including zero.

This demonstrates that improvements observed on one dataset should not automatically be assumed to transfer to every translation domain.

## 14. Limitations

The study has several limitations:

- The main training corpus was restricted to short English sentences.
- The fine-tuning experiment used a subset of the available training data.
- BLEU and chrF do not directly measure perfect semantic equivalence.
- Qualitative translation reviews were preliminary and AI-assisted.
- The original pretrained model's historical training corpus was not fully controlled.
- Exact and normalized dataset separation does not eliminate all possible paraphrase overlap.
- The final held-out test improvements were not accompanied by separate confidence intervals.
- FLORES-200 was evaluated as a single external benchmark, not as evidence of universal performance.
- Small differences in measured inference speed should not be overinterpreted.

## 15. Conclusion

This project investigated English-to-French neural machine translation using traditional baselines, a pretrained Transformer, and continued model fine-tuning.

The pretrained Transformer substantially outperformed the dictionary and TF-IDF retrieval baselines.

Additional training improved translation similarity on the original dataset, with the extended fine-tuned model achieving **55.74 BLEU and 71.58 chrF** on 2,000 held-out test sentences.

External FLORES-200 evaluation showed only modest gains, and paired bootstrap analysis did not establish statistically reliable improvements on that benchmark.

The project therefore demonstrates both the potential benefits of Transformer fine-tuning and the importance of rigorous evaluation, statistical uncertainty analysis, and cross-dataset testing.

## 16. Tools and Reproducibility

Experiments were implemented in Python using Google Colab, PyTorch, Hugging Face Transformers, Hugging Face Datasets, scikit-learn, Pandas, NumPy, and SacreBLEU.

All four experimental notebooks are available in the GitHub repository:

- `01_data_and_baseline.ipynb`
- `02_evaluation_improvements.ipynb`
- `03_finetuning.ipynb`
- `04_external_evaluation.ipynb`

The corresponding predictions, evaluation metrics, and bootstrap results are stored in the `results/` directory.

Fine-tuned model checkpoints were saved separately in Google Drive.
