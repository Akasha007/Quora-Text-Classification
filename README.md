# Quora-Text-Classification

## Text Classification with Bag of Words

> **Bag of Words**: The bag-of-words (BOW) model is a representation that turns arbitrary text into fixed-length vectors by counting how many times each word appears.

### Table of Contents

1. [Download and Explore the Data](#download-and-explore-the-data)
2. [Text Preprocessing Techniques](#text-preprocessing-techniques)
    - [Bag of Words Intuition](#bag-of-words-intuition)
3. [Implement Bag of Words](#implement-bag-of-words)
4. [ML Models for Text Classification](#ml-models-for-text-classification)
    - [Train Logistic Regression model](#train-logistic-regression-model)
    - [Make Predictions using the model](#make-predictions-using-the-model)
5. [Make Predictions and Submit to Kaggle](#make-predictions-and-submit-to-kaggle)

---

### Download and Explore the Data

Download the dataset from [Kaggle](https://www.kaggle.com/c/quora-insincere-questions-classification) to Colab. Explore the data using Pandas. Create a small working sample.

### Text Preprocessing Techniques

Understand the bag of words model, including tokenization, stop word removal, and stemming.

#### Bag of Words Intuition

1. Create a list of all the words across all the text documents.
2. Convert each document into vector counts of each word.

Limitations:
- There may be too many words in the dataset.
- Some words may occur too frequently.
- Some words may occur very rarely or only once.
- A single word may have many forms (go, gone, going or bird vs. birds).

### Implement Bag of Words

1. Create a vocabulary using Count Vectorizer.
2. Transform text to vectors using Count Vectorizer.
3. Configure text preprocessing in Count Vectorizer.

### ML Models for Text Classification

Create a training & validation set. Train a logistic regression model and make predictions on training, validation & test data.

#### Train Logistic Regression model

```python
from sklearn.linear_model import LogisticRegression

MAX_ITER = 1000

model = LogisticRegression(max_iter=MAX_ITER, solver='sag')

%%time
model.fit(train_inputs, train_targets)
```
#### Make Predictions using the model

```python
train_preds = model.predict(train_inputs)

# other evaluation metrics
```

#### Make Predictions and Submit to Kaggle

```python
test_preds = model.predict(test_inputs)

# save predictions to submission.csv
```

