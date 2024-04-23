import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as panda
import seaborn as sns
import re
import nltk

# anograms = {
#     "AFAIK": "As Far As I Know",
#     "AFK": "Away From Keyboard",
#     "ASAP": "As Soon As Possible",
#     "ATK": "At The Keyboard",
#     "ATM": "At The Moment",
#     "A3": "Anytime, Anywhere, Anyplace",
#     "BAK": "Back At Keyboard",
#     "BBL": "Be Back Later",
#     "BBS": "Be Back Soon",
#     "BFN": "Bye For Now",
#     "B4N": "Bye For Now",
#     "BRB": "Be Right Back",
#     "BRT": "Be Right There",
#     "BTW": "By The Way",
#     "B4": "Before",
#     "B4N": "Bye For Now",
#     "CU": "See You",
#     "CUL8R": "See You Later",
#     "CYA": "See You",
#     "FAQ": "Frequently Asked Questions",
#     "FC": "Fingers Crossed",
#     "FWIW": "For What It's Worth",
#     "FYI": "For Your Information",
#     "GAL": "Get A Life",
#     "GG": "Good Game",
#     "GN": "Good Night",
#     "GMTA": "Great Minds Think Alike",
#     "GR8": "Great!",
#     "G9": "Genius",
#     "IC": "I See",
#     "ICQ": "I Seek you (also a chat program)",
#     "ILU": "ILU: I Love You",
#     "IMHO": "In My Honest/Humble Opinion",
#     "IMO": "In My Opinion",
#     "IOW": "In Other Words",
#     "IRL": "In Real Life",
#     "KISS": "Keep It Simple, Stupid",
#     "LDR": "Long Distance Relationship",
#     "LMAO": "Laugh My A.. Off",
#     "LOL": "Laughing Out Loud",
#     "LTNS": "Long Time No See",
#     "L8R": "Later",
#     "MTE": "My Thoughts Exactly",
#     "M8": "Mate",
#     "NRN": "No Reply Necessary",
#     "OIC": "Oh I See",
#     "PITA": "Pain In The A..",
#     "PRT": "Party",
#     "PRW": "Parents Are Watching",
#     "QPSA?": "Que Pasa?",
#     "ROFL": "Rolling On The Floor Laughing",
#     "ROFLOL": "Rolling On The Floor Laughing Out Loud",
#     "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
#     "SK8": "Skate",
#     "STATS": "Your sex and age",
#     "ASL": "Age, Sex, Location",
#     "THX": "Thank You",
#     "TTFN": "Ta-Ta For Now!",
#     "TTYL": "Talk To You Later",
#     "U": "You",
#     "U2": "You Too",
#     "U4E": "Yours For Ever",
#     "WB": "Welcome Back",
#     "WTF": "What The F...",
#     "WTG": "Way To Go!",
#     "WUF": "Where Are You From?",
#     "W8": "Wait...",
#     "7K": "Sick:-D Laugher",
#     "TFW": "That feeling when",
#     "MFW": "My face when",
#     "MRW": "My reaction when",
#     "IFYP": "I feel your pain",
#     "TNTL": "Trying not to laugh",
#     "JK": "Just kidding",
#     "IDC": "I don't care",
#     "ILY": "I love you",
#     "IMU": "I miss you",
#     "ADIH": "Another day in hell",
#     "ZZZ": "Sleeping, bored, tired",
#     "WYWH": "Wish you were here",
#     "TIME": "Tears in my eyes",
#     "BAE": "Before anyone else",
#     "FIMH": "Forever in my heart",
#     "BSAAW": "Big smile and a wink",
#     "BWL": "Bursting with laughter",
#     "BFF": "Best friends forever",
#     "CSL": "Can't stop laughing"
# }
# def replace_chat_words(curr):
#     text = curr.split()
#     for i, next in enumerate(text):
#         if text.lower() in anograms:
#             text[i] = anograms[text.lower()]
#     return ''.join(text)

nltk.download('punkt')
nltk.download('stopwords')
data_set = panda.read_csv('../text.csv')
data_set.head(10)
data_set.columns
data_set.drop(columns='Unnamed: 0', inplace=True)
data_set = data_set.drop_duplicates()


e_map = {5:'Suprise', 4:'Scared', 3:'Anger', 2:'Love', 1:'Happy', 0:'Sad'}
data_set['label'] = data_set['label'].map(e_map)
data_set['text'] = data_set['text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
data_set['text'] = data_set['text'].str.replace(r'\d+', '', regex=True)
data_set['text'] = data_set['text'].str.replace(r'\s+', '', regex=True)
data_set['text'] = data_set['text'].str.replace(r'[^\w\s]', '', regex=True)
data_set.head()


rev_e_map = {'Suprise':5, 'Scared':4, 'Anger':3, 'Love':2, 'Happy':1, 'Sad':0}
data_set['label']=data_set['label'].map(rev_e_map)
data_set.head()


xhold = data_set['text']
yhold = data_set['label']
X_train, X_test, y_train, y_test = train_test_split(xhold, yhold, test_size=0.2, random_state=42)
token = nltk.Tokenize(num_words=60000)
token.fit_on_texts(X_train)
token.fit_on_texts(X_test)
train_seq = token.texts_to_sequence(X_train)
maxi = max(len(token) for token in train_seq)
test_seq = token.texts_to_sequence(X_test)
train_pad = pad_sequences(train_seq, maxi=maxi, padding='post')
size = np.max(train_pad) + 1


# Begining of the model #
RNN_model = Sequential()
RNN_model.add(Embedding(input_dim=size, output_dim = 100, input_shape=(79,)))
RNN_model.add(Bidirectional(LSTM(128)))
RNN_model.add(BatchNormalization())
RNN_model.add(Dropout(0.5))
RNN_model.add(Dense(64, activation='relu'))
RNN_model.add(Dropout(0.5))
RNN_model.add(Dense(6, activation='softmax'))

# Compilation of the model #
RNN_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
data_points = RNN_model.fit(train_pad, y_train, epochs=15, batch_size=32, validation_data=(test_pad, y_train), callbacks=[EarlyStopping(patience=3)])