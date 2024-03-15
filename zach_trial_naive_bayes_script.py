import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
#Load data from CSV file, need to change path and add file to github
data = pd.read_csv('text_data.csv')

#Preprocess the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['emotion']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

#  Evaluate the classifier
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

## attempt at visualization

# Visualize the initial emotion distribution 
plt.figure(figsize=(8, 6))
sns.countplot(data['emotion'])
plt.title('Distribution of Emotions')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.show()


## visualize model

##NEED TO BE Implemented
