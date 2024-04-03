import re

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB

# Downloading necessary NLTK data
nltk.download('punkt')

class EmotionClassifier:
    def __init__(self):
        self.model_w2v = None
        self.model_nb = None

    @staticmethod
    def preprocess_text(text):
        # using text normalization for preprocessing
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", '', text)

        # tokenization for preprocessing
        tokens = word_tokenize(text)
        
        # negation handling for preprocessing
        negation_terms = {'not', 'no', 'never', 'none'}
        tokens = ['not_' + tokens[i+1] if tokens[i] in negation_terms and i+1 < len(tokens) else token 
                  for i, token in enumerate(tokens)]

        return ' '.join(tokens)

    def train_word2vec(self, texts):
        # train a Word2Vec model
        self.model_w2v = Word2Vec(texts, vector_size=100, window=5, min_count=1)

    def tweet_to_vector(self, tweet):
        # convert data to vectors using word2vec model
        words = tweet.split()
        word_vectors = [self.model_w2v.wv[word] for word in words if word in self.model_w2v.wv]
        return np.mean(word_vectors, axis=0) if len(word_vectors) > 0 else np.zeros(100)

    def train(self, X_train, y_train):
        # convert text to vectors and train Naive Bayes model
        self.train_word2vec([text.split() for text in X_train])
        X_train_vect = np.array([self.tweet_to_vector(tweet) for tweet in X_train])

        # Training Naive Bayes
        self.model_nb = GaussianNB()
        self.model_nb.fit(X_train_vect, y_train)

    def predict(self, X):
        # same as training using word2vec model 
        X_vect = np.array([self.tweet_to_vector(tweet) for tweet in X])
        return self.model_nb.predict(X_vect)

# load and preprocess the data
df = pd.read_csv("../text.csv")
df['processed_text'] = df['text'].apply(EmotionClassifier.preprocess_text)

classifier = EmotionClassifier()

def cross_validate_evaluate(X, y):
    # using cross validation 
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    acc_scores, prec_scores, f1_scores = [], [], []
    aggregated_cm = np.zeros((6, 6))

    for train_index, test_index in kf.split(X):
        # split the data into training and testing sets
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