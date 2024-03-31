import re

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB


class EmotionClassifier:
    def __init__(self):
        self.model_nb = MultinomialNB()
        self.vectorizer = CountVectorizer()

    @staticmethod
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", '', text)

        tokens = word_tokenize(text)

        negation_terms = {'not', 'no', 'never', 'none'}
        tokens = ['not_' + tokens[i+1] if tokens[i] in negation_terms and i+1 < len(tokens) else token 
                  for i, token in enumerate(tokens)]

        return ' '.join(tokens)

    def train(self, X_train, y_train):
        X_train_vect = self.vectorizer.fit_transform(X_train)
        self.model_nb.fit(X_train_vect, y_train)

    def predict(self, X):
        X_vect = self.vectorizer.transform(X)
        return self.model_nb.predict(X_vect)


df = pd.read_csv("text.csv")
df['processed_text'] = df['text'].apply(EmotionClassifier.preprocess_text)

classifier = EmotionClassifier()

def cross_validate_evaluate(X, y):
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    acc_scores, prec_scores, f1_scores = [], [], []
    aggregated_cm = np.zeros((6, 6))

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier.train(X_train, y_train)
        y_pred = classifier.predict(X_test)

        # Metrics
        acc_scores.append(accuracy_score(y_test, y_pred))
        prec_scores.append(precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))

        # Aggregate confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
        aggregated_cm += cm

    return acc_scores, prec_scores, f1_scores, aggregated_cm


# Need numpy array for crossvalid 
X = df['processed_text'].to_numpy()
y = df['label'].to_numpy()

acc, prec, f1, cm = cross_validate_evaluate(X, y)

print("Average Accuracy:", np.mean(acc))
print("Average Precision:", np.mean(prec))
print("Average F1-Score:", np.mean(f1))

# plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Aggregated Confusion Matrix')
plt.show()