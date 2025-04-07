#Loading Libraries

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
import string 


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
 

data = {
    'text' : [
        "I love this product!",
        "This is terrible.",
        "Not bad, could be better.",
        "Absolutely fantastic experience.",
        "Worst thing ever.",
        "I'm neutral about this."
    ],
    'sentiment': ['positive','negative','neutral','positive','negative','neutral']
}

df = pd.DataFrame(data)

#preprossing

def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ''.join(tokens)

df['clean_text'] = df['text'].apply(preprocess)

#vectorize

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Training