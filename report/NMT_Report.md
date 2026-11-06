# Neural Machine Translation: English-to-French Translation

## 1. Introduction

Neural Machine Translation (NMT) uses neural networks to translate text from one language into another.

The objective of this project is to translate short English sentences into French using an existing pretrained Transformer model and compare its performance against a simple translation baseline.

## 2. Methodology

### 2.1 Dataset

The project uses an English–French parallel dataset containing pairs of sentences in both languages.

The dataset is cleaned by removing missing values, empty sentences, duplicate sentence pairs, and English sentences longer than 15 words.

The data is divided into training and test sets. A sample of 100 unseen test sentences is selected for evaluation.

### 2.2 Dictionary Baseline

A simple English–French dictionary is implemented as the baseline translation method.

Each English word is replaced by its corresponding French word if it exists in the dictionary. Unknown words remain unchanged.

This approach is computationally simple but cannot reliably handle sentence structure, context, or grammatical differences between languages.

### 2.3 Pretrained Transformer

The project uses the pretrained `Helsinki-NLP/opus-mt-en-fr` model from Hugging Face.

This is a sequence-to-sequence Transformer model designed to translate English text into French.

The tokenizer converts input sentences into token IDs, and the model generates French token sequences that are decoded into readable text.

The pretrained model is used for inference without additional training.

## 3. Evaluation

Both methods are evaluated on the same 100 English sentences using their corresponding French references.

Two evaluation metrics are used:

- **BLEU:** Measures overlap between generated and reference translations using word n-grams and a brevity penalty.
- **chrF:** Measures similarity using character n-grams.

Higher scores generally indicate greater similarity to the reference translations.

## 4. Experimental Results

| Translation Method | BLEU | chrF |
|---|---:|---:|
| Dictionary Baseline | 0.54 | 15.52 |
| Pretrained Transformer | 57.97 | 71.93 |

The pretrained Transformer achieves considerably higher BLEU and chrF scores than the dictionary baseline.

## 5. Discussion and Limitations

The dictionary baseline performs poorly because it contains only a small number of predefined word translations. It cannot generate grammatically appropriate sentences in most cases.

The Transformer produces substantially more fluent and accurate translations.

However, the baseline is deliberately limited, so this comparison does not establish that the Transformer would outperform every alternative translation method.

Furthermore, BLEU and chrF measure reference similarity rather than perfect semantic correctness. Valid translations using different expressions may receive lower scores.

The evaluation is also limited to a sample of 100 short sentences.

## 6. Conclusion

This project demonstrates how an existing pretrained Transformer can be used for English-to-French Neural Machine Translation.

The results show that the pretrained Transformer substantially outperforms the simple dictionary baseline on the selected test set.

The project demonstrates text preprocessing, pretrained model inference, translation generation, and quantitative evaluation using BLEU and chrF.
