# Financial Sentiment Analysis

A 3-class sentiment classification project on financial text (negative / neutral / positive), progressing from classical ML baselines (TF-IDF + Naive Bayes/Random Forest/XGBoost) through a BiLSTM to transformer fine-tuning (DistilBERT, FinBERT), run as a controlled comparison to isolate what actually moves the needle.

---

## Objective

Classify financial text into negative, neutral, or positive sentiment, and compare classical, deep learning, and transformer approaches on the same train/test split to understand the real accuracy/complexity trade-off — not just chase the highest number.

## Dataset

- 45,000 lemmatized financial text samples, pre-cleaned (`preprocessed_data_lemma.csv`)
- 3 classes: negative, neutral, positive — moderately imbalanced (test split: neutral 3,546 / positive 3,240 / negative 2,214 out of 9,000)
- 80/20 train/test split, `random_state=42`
- Zero empty rows after lemmatization; median text length ~89 characters

## Methodology

Every model is scored with the same `record_result()` helper — accuracy, macro F1 (chosen over weighted F1 since class imbalance shouldn't let the majority class dominate the score), a normalized confusion matrix, and a full classification report — so results are directly comparable across the whole pipeline.

### Classical ML Block (TF-IDF features)
`TfidfVectorizer(max_features=2000, min_df=5, max_df=0.8, ngram_range=(1,2))` — unigrams + bigrams, capped vocabulary to control overfitting on a modest dataset size.

- Multinomial Naive Bayes — `GridSearchCV` over alpha (5-fold, scoring=macro F1)
- Random Forest — `RandomizedSearchCV` over n_estimators/max_depth/min_samples_split/min_samples_leaf
- XGBoost — `RandomizedSearchCV` defined with the same TF-IDF features (GPU-accelerated, T4); grid search configured but results weren't captured in this saved run

Each grid search also logs train vs. CV F1 to flag overfitting before accepting a model.

### Deep Learning — BiLSTM
Text tokenized (vocab size 5,000, max sequence length 40, covering the vast majority of observed sentence lengths) and padded, then fed into a Bidirectional LSTM:

```
Embedding → Bidirectional(LSTM) → Dropout(0.3) → Dense(32, relu) → Dense(3, softmax)
```

A small manual sweep over 3 configs (embedding dim, LSTM units, learning rate) was run, keeping the config with the best validation macro F1 rather than a full grid search — training curves showed clear overfitting past a few epochs (train accuracy climbing past 90% while validation loss rose), consistent with a relatively small dataset for a from-scratch sequence model.

### Transformer Fine-Tuning
Two pretrained checkpoints fine-tuned via Hugging Face `Trainer` (3 epochs, same train/test split, same `compute_metrics`):

- `distilbert-base-uncased` — general-purpose baseline
- `yiyanghkust/finbert-tone` — domain-specific financial-sentiment model, run as a controlled ablation (identical epochs and learning rate to DistilBERT) to isolate whether financial-domain pretraining helps on this particular dataset

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Naive Bayes (TF-IDF) | 0.6682 | 0.6648 |
| Random Forest (TF-IDF) | 0.6538 | 0.6418 |
| BiLSTM | 0.5993 | 0.6002 |
| DistilBERT (final) | 0.7854 | 0.7862 |
| FinBERT-tone | 0.7724 | 0.7727 |

DistilBERT is the best-performing model on this dataset and split. The FinBERT ablation, run under identical training conditions, came in slightly below DistilBERT rather than above it — domain-specific pretraining didn't translate into a measurable gain here, which is a useful negative result rather than a failure: it means the dataset's sentiment signal is largely surface-level rather than requiring finance-specific vocabulary understanding, at least at 3 epochs of fine-tuning.

BiLSTM underperforms even the classical TF-IDF baselines — expected for a from-scratch sequence model trained on a comparatively small dataset without pretrained embeddings.

## Key Findings

- Lemmatization was necessary preprocessing but also became an accuracy ceiling for the classical models — TF-IDF + Naive Bayes/Random Forest plateaued in the mid-60s% regardless of hyperparameter tuning.
- Transformer fine-tuning delivers the largest single jump in performance (roughly +12 points macro F1 over the best classical baseline), which is the strongest evidence in this notebook that this task benefits from contextual embeddings over bag-of-words features.
- The FinBERT vs. DistilBERT ablation is the most interesting result: domain-specific pretraining is not a free win — it needs to be tested against a general-purpose baseline under matched conditions rather than assumed.

## Tech Stack

- Data processing: pandas, NumPy
- Classical ML: scikit-learn (TF-IDF, Naive Bayes, Random Forest, GridSearchCV/RandomizedSearchCV), XGBoost
- Deep learning: TensorFlow/Keras (BiLSTM)
- Transformers: Hugging Face `transformers` (DistilBERT, FinBERT), `datasets`, PyTorch backend

## Repository Structure

```
├── financial_sentiment_analysis_og.ipynb   # Full pipeline: classical ML → BiLSTM → transformers
├── preprocessed_data_lemma.csv              # Pre-lemmatized input dataset
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn xgboost tensorflow transformers datasets torch matplotlib seaborn
jupyter notebook financial_sentiment_analysis_og.ipynb
```

GPU is recommended for the BiLSTM and transformer sections (notebook was run on a Tesla T4). Run all cells sequentially — the train/test split defined early in the notebook is reused across every model for a fair comparison.

## Author

Shashank Shekhar — M.Sc. Statistics (Applied Statistics and Informatics), IIT Bombay
