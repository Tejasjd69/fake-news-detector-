Fake News Classifier

An LSTM-based deep learning model that classifies news statements as real or fake using TensorFlow/Keras.

Overview

This project trains a text classification model on a labeled dataset of news statements. Text is cleaned, stemmed, and converted into padded word-embedding sequences, then fed into an LSTM network that predicts whether a given piece of news is likely fake.

Dataset
Input file: fkenews.csv
Relevant columns used: Statement, Web, Label
Dropped columns: Image, Date, id, Category
Label: 0 = real, 1 = fake
The raw dataset is imbalanced (37,800 real vs. 18,914 fake). It is balanced via random undersampling of the majority (real) class to match the number of fake samples.
Statement and Web are concatenated into a single content field used as model input.
Pipeline
Data loading & cleaning — read CSV, drop unused columns, balance classes.
Text preprocessing — for each content entry:
Strip non-alphabetic characters
Lowercase and tokenize
Remove English stopwords (NLTK)
Apply Porter stemming
Vectorization — one-hot encode words into a 10,000-word vocabulary space, then pad/truncate all sequences to a fixed length of 20 tokens.
Model architecture:
Embedding layer (vocab size 10,000 → 40-dim vectors)
Dropout (0.3)
LSTM (100 units)
Dropout (0.3)
Dense output layer with sigmoid activation (binary classification)
Loss: binary_crossentropy, Optimizer: adam
Training — 67/33 train-test split, trained for 10 epochs, batch size 64.
Evaluation — confusion matrix and accuracy score on the held-out test set.
Inference — a predict_news(text) helper preprocesses raw text and returns True if the model predicts the news is fake.
Model persistence — the trained model is saved both via pickle (news_classifier.sav) and Keras's native format (.keras).
Results
Metric	Value
Validation accuracy (final epoch)	~98.9%
Test set accuracy	~98.8%
Confusion matrix	[[6078, 42], [110, 6254]]
Requirements
Python 3
pandas, numpy
scikit-learn
tensorflow (Keras)
nltk (with stopwords corpus downloaded)

Install dependencies:

bash
pip install pandas numpy scikit-learn tensorflow nltk
Usage
Place fkenews.csv in the expected path (update the pd.read_csv path if not using Google Colab's /content/ directory).
Run the notebook fakenewsclass.ipynb top to bottom to preprocess data, train the model, and evaluate it.
Use the model for inference:
python
text = "india can never overtake china"
is_fake = predict_news(text)
print("The news is likely to be fake" if is_fake else "The news is likely to be real")
To reload a saved model later:
python
from tensorflow.keras.models import load_model
model = load_model("path/to/location.keras")
