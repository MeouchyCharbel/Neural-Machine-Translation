# Neural Machine Translation — English to French

## Project Overview

This NLP project translates short English sentences into French using a pretrained Transformer model.

The project compares two translation approaches:

1. **Dictionary Baseline:** A simple word-by-word translation method using a predefined English–French dictionary.
2. **Pretrained Transformer:** The Hugging Face `Helsinki-NLP/opus-mt-en-fr` model.

Both methods are evaluated using BLEU and chrF scores.

## Technologies Used

- Python
- Google Colab
- Hugging Face Transformers
- PyTorch
- pandas
- scikit-learn
- SacreBLEU
- Matplotlib

## Dataset

The project uses an English–French parallel sentence dataset downloaded from the TensorFlow dataset repository.

The data is cleaned by removing missing values, empty sentences, duplicate sentence pairs, and sentences exceeding 15 English words.

A separate test set is created, and 100 sentences are selected for evaluation.

## How to Run

1. Open `01_data_and_baseline.ipynb` in Google Colab.
2. Run the notebook cells in order.
3. The notebook downloads and prepares the dataset.
4. The dictionary baseline generates translations.
5. The pretrained Transformer is downloaded and loaded.
6. Both methods translate the same 100 test sentences.
7. BLEU and chrF scores are calculated and visualized.
8. Enter a new English sentence in the final input cell to generate its French translation.

The required Python dependencies are listed in `requirements.txt`.

## Evaluation Results

| Model | BLEU | chrF |
|---|---:|---:|
| Dictionary Baseline | 0.54 | 15.52 |
| Pretrained Transformer | 57.97 | 71.93 |

The pretrained Transformer achieves substantially higher scores than the dictionary baseline on the selected test set.

The baseline is deliberately simple and has a limited vocabulary, so these results should not be interpreted as a comparison against all traditional translation systems.

## Repository Structure

```text
Neural-Machine-Translation/
├── 01_data_and_baseline.ipynb
├── data/
│   └── sample_pairs.csv
├── notebooks/
├── results/
│   ├── metrics.csv
│   └── translation_comparison.csv
├── report/
│   └── NMT_Report.md
├── requirements.txt
└── README.md
```

## Project Limitations

- The dictionary baseline has limited vocabulary and does not handle grammar reliably.
- The evaluation uses 100 short sentences.
- BLEU and chrF measure similarity to reference translations, not complete semantic accuracy.
- The Transformer is pretrained; no additional model training or fine-tuning is performed.

## Conclusion

The project demonstrates how a pretrained Transformer can be used for English-to-French Neural Machine Translation and evaluated against a simple baseline using standard translation metrics.
