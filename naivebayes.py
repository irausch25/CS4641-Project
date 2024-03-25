import pandas as pd
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from gensim.models import Word2Vec

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

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return classification_report(y_test, y_pred)


# Load dataset
df = pd.read_csv("sample.csv")
df['processed_text'] = df['text'].apply(EmotionClassifier.preprocess_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['label'], test_size=0.2)

classifier = EmotionClassifier()
classifier.train(X_train, y_train)

# Evaluation
print(classifier.evaluate(X_test, y_test))

