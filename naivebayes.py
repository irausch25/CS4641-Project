import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, accuracy_score, precision_score, f1_score, confusion_matrix
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns

class EmotionClassifier:
    def __init__(self):
        self.model_w2v = None
        self.model_nb = None

    @staticmethod
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", '', text)

        tokens = word_tokenize(text)

        negation_terms = {'not', 'no', 'never', 'none'}
        tokens = ['not_' + tokens[i+1] if tokens[i] in negation_terms and i+1 < len(tokens) else token 
                  for i, token in enumerate(tokens)]

        return ' '.join(tokens)

    def train_word2vec(self, texts):
        self.model_w2v = Word2Vec(texts, vector_size=100, window=5, min_count=1)

    def tweet_to_vector(self, tweet):
        words = tweet.split()
        word_vectors = [self.model_w2v.wv[word] for word in words if word in self.model_w2v.wv]
        return np.mean(word_vectors, axis=0) if len(word_vectors) > 0 else np.zeros(100)

    def train(self, X_train, y_train):
        self.train_word2vec([text.split() for text in X_train])

        X_train_vect = np.array([self.tweet_to_vector(tweet) for tweet in X_train])

        # Training Naive Bayes
        self.model_nb = GaussianNB()
        self.model_nb.fit(X_train_vect, y_train)

    def predict(self, X):
        X_vect = np.array([self.tweet_to_vector(tweet) for tweet in X])
        return self.model_nb.predict(X_vect)


df = pd.read_csv("text.csv")
df['processed_text'] = df['text'].apply(EmotionClassifier.preprocess_text)

classifier = EmotionClassifier()

def cross_validate_evaluate(X, y):
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    acc_scores, prec_scores, f1_scores = [], [], []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier.train(X_train, y_train)
        y_pred = classifier.predict(X_test)

        # Metrics
        acc_scores.append(accuracy_score(y_test, y_pred))
        prec_scores.append(precision_score(y_test, y_pred, average='weighted'))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

    return acc_scores, prec_scores, f1_scores

# Need numpy array for crossvalid 
X = df['processed_text'].to_numpy()
y = df['label'].to_numpy()

acc, prec, f1 = cross_validate_evaluate(X, y)

print("Average Accuracy:", np.mean(acc))
print("Average Precision:", np.mean(prec))
print("Average F1-Score:", np.mean(f1))
