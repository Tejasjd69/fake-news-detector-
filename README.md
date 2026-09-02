# 📰 Fake News Classifier

[![Python](https://img.shields.io/badge/Python-3.x-blue)]()
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-orange)]()
[![Model](https://img.shields.io/badge/Model-LSTM-green)]()
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.8%25-brightgreen)]()

An LSTM-based deep learning model that classifies news statements as **Real** or **Fake** using TensorFlow/Keras.

---

## 📑 Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Pipeline](#pipeline)
4. [Model Architecture](#model-architecture)
5. [Results](#results)
6. [Requirements](#requirements)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Project Structure](#project-structure)
10. [Known Issues & Future Work](#known-issues--future-work)
11. [License](#license)

---

## Overview

| | |
|---|---|
| **Task** | Binary text classification |
| **Domain** | News / misinformation detection |
| **Approach** | NLP preprocessing + word embeddings + LSTM |
| **Output** | Probability that a statement is fake news |

This project trains a text classification model on a labeled dataset of news statements. Text is cleaned, stemmed, and converted into padded word-embedding sequences, then fed into an LSTM network that predicts whether a given piece of news is likely fake.

---

## Dataset

| Attribute | Detail |
|---|---|
| Source file | `fkenews.csv` |
| Columns used | `Statement`, `Web`, `Label` |
| Columns dropped | `Image`, `Date`, `id`, `Category` |
| Label encoding | `0` = Real, `1` = Fake |
| Original class balance | 37,800 real / 18,914 fake |
| Balancing method | Random undersampling of majority class |
| Balanced class size | 18,914 / 18,914 |
| Model input | `content` = `Statement` + `Web` (concatenated) |

---

## Pipeline

| Step | Description |
|---|---|
| 1. Load & clean | Read CSV, drop unused columns |
| 2. Balance classes | Undersample majority class (`sklearn.utils.resample`) |
| 3. Merge text fields | Concatenate `Statement` + `Web` into `content` |
| 4. Clean text | Remove non-alphabetic characters, lowercase |
| 5. Remove stopwords | NLTK English stopword list |
| 6. Stem words | Porter Stemmer |
| 7. Vectorize | Keras `one_hot` encoding, vocab size 10,000 |
| 8. Pad sequences | Fixed length of 20 tokens (`padding='pre'`) |
| 9. Train/test split | 67% train / 33% test |
| 10. Train model | 10 epochs, batch size 64 |
| 11. Evaluate | Confusion matrix + accuracy score |
| 12. Save model | `.sav` (pickle) and `.keras` formats |

---

## Model Architecture

| Layer | Type | Output Shape | Parameters | Notes |
|---|---|---|---|---|
| 1 | Embedding | `(None, 20, 40)` | 400,000 | vocab_size=10,000, embedding_dim=40 |
| 2 | Dropout | `(None, 20, 40)` | 0 | rate=0.3 |
| 3 | LSTM | `(None, 100)` | 56,400 | units=100 |
| 4 | Dropout | `(None, 100)` | 0 | rate=0.3 |
| 5 | Dense | `(None, 1)` | 101 | activation=sigmoid |

**Compilation settings:**

| Setting | Value |
|---|---|
| Loss function | `binary_crossentropy` |
| Optimizer | `adam` |
| Metric | `accuracy` |

---

## Results

| Metric | Value |
|---|---|
| Training accuracy (final epoch) | 99.9% |
| Validation accuracy (final epoch) | 98.9% |
| **Test accuracy** | **98.8%** |

**Confusion Matrix (Test Set):**

| | Predicted Real | Predicted Fake |
|---|---|---|
| **Actual Real** | 6,078 | 42 |
| **Actual Fake** | 110 | 6,254 |

---

## Requirements

| Package | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Array operations |
| `scikit-learn` | Resampling, train/test split, metrics |
| `tensorflow` | Keras model building and training |
| `nltk` | Stopword removal, stemming |

---

## Installation

```bash
pip install pandas numpy scikit-learn tensorflow nltk
```

```python
import nltk
nltk.download('stopwords')
```

---

## Usage

**1. Prepare the dataset**

Place `fkenews.csv` in the expected path. Update the `pd.read_csv` path if not running in Google Colab (default path used: `/content/fkenews.csv`).

**2. Train the model**

Run `fakenewsclass.ipynb` top to bottom to preprocess data, train the model, and evaluate results.

**3. Run inference**

```python
text = "india can never overtake china"
is_fake = predict_news(text)
print("The news is likely to be fake" if is_fake else "The news is likely to be real")
```

**4. Reload a saved model**

```python
from tensorflow.keras.models import load_model
model = load_model("path/to/location.keras")
```

---

## Project Structure

| File | Description |
|---|---|
| `fakenewsclass.ipynb` | Main notebook: preprocessing, training, evaluation |
| `fkenews.csv` | Raw dataset (input) |
| `news_classifier.sav` | Trained model (pickle format) |
| `newsclassi.keras` / `location.keras` | Trained model (Keras native format) |

---

## Known Issues & Future Work

| Issue | Recommendation |
|---|---|
| Hardcoded Windows save path (`C:/Users/msing/Documents/...`) | Replace with a relative or configurable path |
| `one_hot` uses word hashing, not a learned vocabulary | Replace with Keras `Tokenizer` for robust, collision-free vocab handling |
| English-only stopword support | Add multilingual support if needed |
| No cross-validation | Add k-fold cross-validation for more robust evaluation |
| Fixed sequence length (20 tokens) | Analyze statement length distribution to tune `sent_length` |

---

## License

Add a license of your choice here (e.g., MIT, Apache 2.0).
