# Neural Machine Translation: English-to-French Translation Using LSTM and Transformer Models

## 1. Introduction

Neural Machine Translation (NMT) uses artificial neural networks to translate text from one language into another. Unlike traditional word-by-word translation, neural models learn linguistic patterns from parallel sentence data and generate translations based on the source sentence.

This project investigates English-to-French machine translation using five approaches:

1. Dictionary-based translation
2. TF-IDF retrieval-based translation
3. LSTM encoder–decoder without attention
4. LSTM encoder–decoder with attention
5. Pretrained Transformer (MarianMT)

The project is organized into three notebooks. The first introduces basic translation methods, the second improves the evaluation procedure and adds a retrieval baseline, and the third develops and compares the neural models.

The **main final experiment** compares three neural translation systems on the same **2,000 held-out test sentences**.

### 1.1 Research Question

How does adding attention affect the translation quality of an LSTM encoder–decoder model, and how do both LSTM variants compare with a pretrained Transformer?

### 1.2 Project Objectives

- Prepare and preprocess an English–French parallel dataset.
- Implement a dictionary-based translation baseline.
- Develop a TF-IDF retrieval-based translation system.
- Evaluate an existing pretrained Transformer.
- Compare Transformer greedy decoding with beam search.
- Implement an LSTM encoder–decoder without attention.
- Implement an LSTM encoder–decoder with attention.
- Train the two LSTM models under matching experimental settings.
- Evaluate the three neural models using identical test sentences.
- Measure translation quality using BLEU and chrF.
- Analyze translation examples, model performance, and limitations.

---

## 2. Project Organization

The project consists of three Jupyter notebooks.

### 2.1 Notebook 01 — Data and Baseline

**File:** `01_data_and_baseline.ipynb`

This notebook introduces the translation task.

Its main operations include:

- Loading English–French sentence pairs.
- Cleaning and preparing text.
- Implementing a simple dictionary-based translator.
- Loading the pretrained MarianMT Transformer.
- Generating French translations.
- Evaluating translation quality using BLEU and chrF.

Notebook 01 provides an introductory comparison between dictionary-based translation and a pretrained Transformer.

### 2.2 Notebook 02 — Evaluation Improvements

**File:** `02_evaluation_improvements.ipynb`

This notebook improves the evaluation methodology.

Its main operations include:

- Creating training, validation, and test partitions.
- Checking for overlapping examples between partitions.
- Implementing a TF-IDF retrieval baseline.
- Evaluating the dictionary translator, TF-IDF retrieval, and MarianMT on 500 validation sentences.
- Comparing greedy decoding with beam search.
- Computing BLEU, chrF, and inference times.
- Examining translation errors.

**Notebook 02 evaluates three approaches: Dictionary, TF-IDF, and MarianMT. It does not implement or evaluate an LSTM model.**

### 2.3 Notebook 03 — Three-Model Neural Comparison

**File:** `03_lstm_three_model_comparison.ipynb`

This notebook contains the main neural-model experiment.

It:

- Prepares the English–French dataset.
- Creates the training, validation-monitor, and held-out test samples.
- Implements an LSTM without attention.
- Implements an LSTM with attention.
- Trains both LSTM models.
- Loads the pretrained MarianMT Transformer.
- Generates translations for the same 2,000 test sentences.
- Calculates BLEU and chrF for all three models.
- Compares performance and examines translation examples.
- Saves model checkpoints, predictions, and evaluation results.

---

## 3. Dataset and Preprocessing

### 3.1 Dataset Description

The project uses an English–French parallel corpus from the TensorFlow-hosted `fra-eng.zip` dataset.

Each observation contains:

- An English sentence.
- Its corresponding French reference translation.

The original dataset contained **167,130 sentence pairs**.

After cleaning, **166,340 sentence pairs** remained.

### 3.2 Data Preprocessing

The preprocessing pipeline includes:

1. Removing missing and empty sentences.
2. Removing duplicate English–French translation pairs.
3. Filtering English sentences to a maximum of 15 words.
4. Normalizing text for overlap detection.
5. Grouping related sentence pairs to reduce data leakage.
6. Dividing the data into training, validation, and test partitions.

Normalized forms of English and French sentences are used to identify related examples that should not be distributed across different dataset partitions.

### 3.3 Dataset Splitting

A group-aware splitting strategy was used.

| Partition | Sentence pairs |
|---|---:|
| Training | 133,017 |
| Validation | 16,616 |
| Test | 16,707 |
| **Total** | **166,340** |

The implemented leakage checks passed.

For Notebook 03, the following subsets were selected:

| Purpose | Sentence pairs |
|---|---:|
| Training each LSTM | 50,000 |
| Validation monitoring | 1,000 |
| Final testing | 2,000 |

The two LSTM models were trained using the same 50,000 training examples.

The validation-monitor examples were used to select the best trained checkpoints.

The 2,000 final test sentences were selected from the separate held-out test partition and were not used for training or checkpoint selection.

This methodology helps provide an unbiased local comparison of the trained models on previously unseen examples from the project corpus.

---

## 4. Translation Methods

### 4.1 Dictionary-Based Translation

The dictionary baseline translates English sentences using a small predefined English–French word mapping.

The system attempts to translate individual words without modeling the complete sentence context.

**Advantages:**

- Simple to implement.
- Computationally inexpensive.
- Useful as an introductory baseline.

**Limitations:**

- Limited vocabulary.
- No contextual understanding.
- No grammatical agreement modeling.
- Difficulty translating unknown words or expressions.

The dictionary baseline is not intended to compete with modern neural translation systems.

### 4.2 TF-IDF Retrieval-Based Translation

TF-IDF stands for Term Frequency–Inverse Document Frequency.

The TF-IDF system represents English sentences as numerical vectors.

For each new English sentence, the system:

1. Converts the sentence into a TF-IDF vector.
2. Computes cosine similarity against English training sentences.
3. Selects the closest matching training sentence.
4. Returns the French translation associated with that sentence.

The retrieval system indexed **92,892 unique English training sentences**.

Unlike neural machine translation, TF-IDF does not generate a new sentence word by word. It retrieves a translation that already exists in the training dataset.

This approach provides a stronger traditional baseline than the small dictionary.

### 4.3 LSTM Encoder–Decoder Without Attention

LSTM stands for Long Short-Term Memory.

An LSTM is a recurrent neural network designed to process sequences while retaining useful information across multiple time steps.

The first neural model contains:

**Encoder:** A bidirectional LSTM processes the English input and produces representations of the sentence.

**Decoder:** An LSTM generates the French translation one token at a time.

**Embeddings:** Trainable embedding layers represent English and French subword tokens as numerical vectors.

**State initialization:** The decoder is initialized using information from the encoder's final states.

The model does not use a cross-attention mechanism to directly access the entire sequence of encoder outputs during decoding.

Consequently, information from the input sentence must be represented through a more restricted encoder-to-decoder connection.

The model was trained from scratch on 50,000 English–French sentence pairs.

**Trainable parameters:** 5,667,184.

### 4.4 LSTM Encoder–Decoder With Attention

The second model uses the same general encoder–decoder architecture but introduces a cross-attention mechanism.

The model contains:

- A bidirectional LSTM encoder.
- An autoregressive LSTM decoder.
- Separate English and French embedding layers.
- A four-head cross-attention mechanism.
- A final output layer for predicting French tokens.

The attention mechanism allows the decoder to incorporate information from different encoder output positions while generating the translation.

This provides access to a richer representation of the English sentence than relying only on the encoder's final states.

The model was trained from scratch using the same training data and principal hyperparameters as the LSTM without attention.

**Trainable parameters:** 7,466,352.

The difference in parameter counts means that the experiment compares two related architectures but is not a strictly equal-parameter attention ablation.

### 4.5 Pretrained Transformer — MarianMT

The Transformer used in this project is:

`Helsinki-NLP/opus-mt-en-fr`

MarianMT is a pretrained encoder–decoder Transformer designed for English-to-French machine translation.

Unlike recurrent models, Transformers rely on attention mechanisms to represent relationships between tokens.

The MarianMT model was used directly for inference without additional fine-tuning.

This is an important distinction because MarianMT benefited from extensive external pretraining, whereas the two LSTM models were trained from scratch using the project's 50,000 training pairs.

Therefore, the comparison with MarianMT evaluates the performance of the implemented translation systems rather than isolating the effects of architecture alone.

---

## 5. LSTM Training Procedure

### 5.1 Training Configuration

Both LSTM variants used matching training settings.

| Parameter | Value |
|---|---|
| Training examples | 50,000 |
| Validation-monitor examples | 1,000 |
| Maximum epochs | 5 |
| Batch size | 64 |
| Embedding dimension | 256 |
| Hidden dimension | 256 |
| English vocabulary size | 6,000 |
| French vocabulary size | 6,000 |
| Maximum sequence length | 48 tokens |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Loss function | Cross-entropy |
| Early-stopping patience | 2 epochs |
| Random seed | 42 |
| Training hardware | Tesla T4 GPU |

SentencePiece subword tokenization was used for both languages.

### 5.2 Training Process

The training procedure consisted of the following operations:

1. Tokenizing English and French sentence pairs.
2. Converting tokens into numerical identifiers.
3. Creating batches of padded sequences.
4. Passing English sentences through the encoder.
5. Generating French-token predictions using the decoder.
6. Computing cross-entropy loss against the reference French tokens.
7. Updating model parameters with the AdamW optimizer.
8. Monitoring validation loss after each epoch.
9. Saving the checkpoint with the lowest monitoring loss.

Teacher forcing was used during training.

Gradient clipping was also applied to improve training stability.

During inference, the models generated translations autoregressively without receiving reference French tokens.

### 5.3 LSTM Without Attention — Training Results

| Epoch | Training loss | Monitoring loss |
|---|---:|---:|
| 1 | 3.9851 | 3.3091 |
| 2 | 2.9048 | 2.8827 |
| 3 | 2.4363 | 2.6530 |
| 4 | 2.1040 | 2.5249 |
| 5 | 1.8421 | 2.4552 |

The training loss decreased over the five epochs.

The monitoring loss also decreased consistently, reaching its best value at epoch 5.

**Best epoch:** 5

**Best monitoring loss:** 2.4552

The epoch 5 checkpoint was selected for final evaluation.

### 5.4 LSTM With Attention — Training Results

| Epoch | Training loss | Monitoring loss |
|---|---:|---:|
| 1 | 3.3393 | 2.3983 |
| 2 | 1.8272 | 1.9491 |
| 3 | 1.3153 | 1.7818 |
| 4 | 1.0181 | 1.7427 |
| 5 | 0.8101 | 1.7467 |

The training loss decreased across all five epochs.

The monitoring loss improved until epoch 4, then increased slightly at epoch 5.

**Best epoch:** 4

**Best monitoring loss:** 1.7427

The epoch 4 checkpoint was selected for final evaluation.

Early stopping was configured for both LSTMs, but neither experiment triggered it before the five planned epochs were completed.

### 5.5 Training Interpretation

The attention-based LSTM achieved a substantially lower monitoring loss than the LSTM without attention.

This result suggests that using encoder representations through cross-attention helped the model learn English-to-French translation under the tested conditions.

However, training loss and monitoring loss alone are not sufficient to assess translation quality.

The final evaluation therefore uses separate held-out test sentences and corpus-level translation metrics.

---

## 6. Evaluation Metrics

### 6.1 BLEU

BLEU stands for Bilingual Evaluation Understudy.

BLEU evaluates translations using word n-gram overlap between generated outputs and reference translations.

It also applies a brevity penalty to discourage excessively short translations.

A higher BLEU score generally indicates greater agreement with the reference translations.

However, BLEU does not directly guarantee semantic correctness.

### 6.2 chrF

chrF is a character n-gram F-score.

It measures the precision and recall of character sequences in generated translations compared with reference sentences.

It can capture partial similarities between words and morphological forms.

A higher chrF score generally indicates greater character-level agreement with the references.

### 6.3 Evaluation Limitations

Both BLEU and chrF are reference-based metrics.

A translation may express the correct meaning using different wording and still receive a lower score.

Therefore, BLEU and chrF should be interpreted as indicators of translation quality rather than direct accuracy percentages.

Qualitative inspection of translated sentences provides useful complementary information.

---

## 7. Experimental Results

### 7.1 Notebook 01 — Initial Baseline Comparison

Notebook 01 evaluated the dictionary translator and pretrained Transformer on an initial sample of 100 sentences.

| Model | BLEU | chrF |
|---|---:|---:|
| Dictionary baseline | 0.54 | 15.52 |
| MarianMT Transformer | 57.97 | 71.93 |

The pretrained Transformer substantially outperformed the dictionary baseline.

The dictionary's limited vocabulary and lack of context contributed to its weak performance.

This was an introductory experiment using a different evaluation sample from the later experiments.

### 7.2 Notebook 02 — Improved Baseline Comparison

Notebook 02 evaluated **three translation approaches** using the same 500 validation sentences.

| Model | BLEU | chrF |
|---|---:|---:|
| Dictionary baseline | 0.66 | 16.14 |
| TF-IDF retrieval | 18.39 | 37.74 |
| MarianMT Transformer | 55.11 | 70.40 |

TF-IDF substantially improved over the dictionary baseline by retrieving translations associated with similar English training sentences.

MarianMT achieved the highest BLEU and chrF scores.

**No LSTM models were included in Notebook 02.**

### 7.3 Notebook 02 — Transformer Decoding Comparison

Notebook 02 also investigated three decoding strategies for MarianMT.

| Decoding strategy | BLEU | chrF |
|---|---:|---:|
| Greedy decoding | 55.11 | 70.40 |
| Beam search — 3 beams | 55.19 | 70.39 |
| Beam search — 5 beams | 54.87 | 70.12 |

Beam search with three beams produced a very small BLEU increase over greedy decoding.

Beam search with five beams produced slightly lower scores.

Overall, beam search did not provide a substantial translation-quality improvement on this validation sample.

Therefore, greedy decoding was selected for the final neural-model comparison.

### 7.4 Notebook 03 — Final Three-Model Comparison

The final experiment evaluates three neural translation models using **2,000 identical held-out test sentences**.

All three models receive the same English inputs and are evaluated against the same French reference translations.

Greedy decoding is used for each model.

| Model | BLEU | chrF |
|---|---:|---:|
| LSTM without attention | 10.46 | 28.24 |
| LSTM with attention | 31.04 | 51.10 |
| **Pretrained MarianMT Transformer** | **53.74** | **68.69** |

These results are saved in:

`results/three_model_test_metrics.csv`

The individual model predictions are saved in:

`results/three_model_test_predictions.csv`

### 7.5 Effect of Attention on the LSTM

The LSTM with attention achieved higher scores than the LSTM without attention.

The differences were:

- **+20.58 BLEU points**
- **+22.86 chrF points**

Both LSTMs used the same training sentences, vocabulary configuration, number of epochs, optimizer, and major training hyperparameters.

This supports the conclusion that the attention-based implementation performed substantially better than the non-attention implementation under the tested conditions.

Nevertheless, the attention mechanism also introduced additional trainable parameters.

The experiment therefore does not completely isolate attention from differences in model capacity.

### 7.6 Comparison With the Pretrained Transformer

MarianMT achieved the highest BLEU and chrF scores in the final experiment.

Compared with the LSTM with attention, MarianMT achieved:

- **+22.70 BLEU points**
- **+17.59 chrF points**

This indicates that MarianMT produced translations with the highest reference-based agreement among the three tested systems.

However, MarianMT was already pretrained, while the two LSTMs were trained from scratch.

The performance gap therefore cannot be explained by architecture alone.

Differences in training data, pretraining, model capacity, and optimization also influence the comparison.

### 7.7 Interpretation of the Results

The final test results show the following ordering:

**LSTM without attention → LSTM with attention → Pretrained MarianMT**

The attention-based LSTM achieved substantially higher scores than its non-attention counterpart.

The pretrained Transformer achieved the strongest overall translation quality.

These findings demonstrate both the benefit of attention in the tested LSTM models and the strong performance of an existing pretrained translation system.

The score differences are descriptive findings from the experiment. No formal statistical-significance test was performed for the final three-model comparison.

---

## 8. Qualitative Translation Analysis

Numerical evaluation provides an overall comparison, but individual translations help explain differences between the systems.

Notebook 03 displays randomly selected translations and examples where the LSTM without attention achieved low sentence-level chrF.

### 8.1 Example: Translating an Occupation

**English input:**

Tom is now a businessman.

**French reference:**

Tom est maintenant homme d'affaire.

| Model | Translation |
|---|---|
| LSTM without attention | Tom est un peu fatigué. |
| LSTM with attention | Tom est maintenant un homme. |
| MarianMT | Tom est maintenant un homme d'affaires. |

The LSTM without attention incorrectly changed the sentence's meaning.

The attention-based LSTM preserved part of the original meaning but omitted the occupation.

MarianMT produced a translation closely matching the intended meaning.

### 8.2 Example: Expressing Surprise

**English input:**

It came as a surprise.

**French reference:**

C'est arrivé par surprise.

| Model | Translation |
|---|---|
| LSTM without attention | Ça a un peu mal. |
| LSTM with attention | C'est comme une surprise. |
| MarianMT | C'est venu comme une surprise. |

The non-attention LSTM produced a translation that did not preserve the original meaning.

The attention-based LSTM captured the idea of a surprise more effectively.

MarianMT also preserved the general meaning using alternative wording.

This example demonstrates why automatic scores should be complemented by qualitative inspection: different valid or partially valid phrasings do not necessarily match the reference exactly.

### 8.3 Common Translation Difficulties

The experiments illustrate several potential sources of translation errors:

- **Vocabulary limitations:** Simple dictionary systems cannot handle many unknown words.
- **Retrieval errors:** TF-IDF may retrieve a similar sentence with a different meaning.
- **Information loss:** An LSTM without attention may fail to preserve important details from the English input.
- **Incomplete translations:** Neural systems may omit details or select incorrect vocabulary.
- **Reference variation:** Acceptable translations may receive lower scores when their wording differs from the reference.

Qualitative inspection helps identify errors that corpus-level metrics alone cannot explain.

---

## 9. Project Limitations

### 9.1 Unequal Pretraining Conditions

MarianMT benefited from substantial external pretraining, while both LSTMs were trained from scratch using the project dataset.

The comparison therefore does not isolate architecture alone.

### 9.2 Model Capacity Differences

The LSTM with attention contains more trainable parameters than the LSTM without attention.

Consequently, the observed gain cannot be attributed exclusively to attention independently of parameter count.

### 9.3 Limited LSTM Training Data

Each LSTM was trained using 50,000 sentence pairs rather than the entire training partition.

Training on more data or using different hyperparameters could change the final results.

### 9.4 Evaluation Scope

The final experiment uses 2,000 held-out sentences sampled from one English–French parallel corpus.

The results may differ on longer sentences, specialized terminology, or texts from other domains.

### 9.5 Automatic Metric Limitations

BLEU and chrF measure overlap with reference translations.

They do not fully evaluate semantic correctness, fluency, or grammatical acceptability.

### 9.6 Historical Pretraining Data

Although the project checks for overlap between its local training, validation, and test partitions, overlap between test sentences and MarianMT's external pretraining data cannot be definitively ruled out.

### 9.7 Statistical Uncertainty

The final comparison reports BLEU and chrF scores for a single experimental evaluation.

No confidence intervals or formal statistical-significance tests were computed for the new three-model test.

Therefore, the differences should be interpreted as observed experimental results rather than statistically established effects.

### 9.8 Computational Comparison

Inference times depend on factors such as hardware, implementation, model size, batch size, and decoding configuration.

Although the notebook records inference times, the models have different computational requirements.

These timing results should not be interpreted as a controlled comparison of architectural efficiency.

---

## 10. Technologies and Reproducibility

### 10.1 Technologies Used

The project was implemented using:

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

### 10.2 Main Project Files

| File | Purpose |
|---|---|
| `01_data_and_baseline.ipynb` | Initial dictionary and MarianMT experiments |
| `02_evaluation_improvements.ipynb` | TF-IDF, dataset splitting, and decoding evaluation |
| `03_lstm_three_model_comparison.ipynb` | LSTM training and final three-model comparison |
| `fra.txt` | English–French parallel corpus |
| `requirements.txt` | Project dependencies |
| `README.md` | Repository documentation |

### 10.3 Final Result Files

| File | Contents |
|---|---|
| `three_model_test_metrics.csv` | Final BLEU and chrF results on 2,000 test sentences |
| `three_model_test_predictions.csv` | English inputs, French references, and three-model predictions |
| `lstm_training_history.csv` | Training and monitoring losses for both LSTM models |
| `three_model_test_metadata.json` | Experimental configuration and dataset statistics |

These files are stored in the `results/` directory.

The trained LSTM checkpoints and tokenization resources were also saved separately to Google Drive for future use.

---

## 11. Conclusion

This project investigated English-to-French machine translation through a progression from simple translation baselines to neural sequence-to-sequence models.

Notebook 01 introduced dictionary-based translation and a pretrained Transformer.

Notebook 02 improved the evaluation methodology, introduced TF-IDF retrieval, and compared MarianMT decoding strategies.

Notebook 03 implemented two LSTM encoder–decoder models, one without attention and another with attention, and compared them with pretrained MarianMT on **2,000 held-out test sentences**.

The final experimental results were:

| Model | BLEU | chrF |
|---|---:|---:|
| LSTM without attention | 10.46 | 28.24 |
| LSTM with attention | 31.04 | 51.10 |
| MarianMT Transformer | **53.74** | **68.69** |

Adding attention substantially improved the performance of the tested LSTM implementation.

The pretrained MarianMT Transformer obtained the highest translation-quality scores.

However, because MarianMT benefited from external pretraining, the comparison cannot establish that its performance advantage results entirely from architectural differences.

Overall, the project demonstrates the importance of attention mechanisms, training data, pretraining, appropriate dataset separation, and consistent evaluation in Neural Machine Translation.

### Future Improvements

Potential extensions include:

- Training the LSTMs using more sentence pairs.
- Experimenting with additional architectures and hyperparameters.
- Evaluating performance on an external translation dataset.
- Performing paired statistical tests or bootstrap confidence intervals.
- Conducting a more systematic human evaluation of translation quality.
- Studying the impact of attention using more closely matched model capacities.
