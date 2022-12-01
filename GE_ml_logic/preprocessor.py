from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import unidecode
import nltk
import pandas as pd
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from ml_logic.params import TARGET_COLUMN








def clean(sentence):

    # Basic cleaning
    sentence = sentence.strip() ## remove whitespaces
    sentence = sentence.lower() ## lowercase
    sentence = ''.join(char for char in sentence if not char.isdigit()) ## remove numbers

    # Advanced cleaning
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '') ## remove punctuation

    unaccented_string = unidecode.unidecode(sentence) # remove accents

    tokenized_sentence = word_tokenize(unaccented_string) ## tokenize
    stop_words = set(stopwords.words('english')) ## define stopwords

    tokenized_sentence_cleaned = [ ## remove stopwords
    w for w in tokenized_sentence if not w in stop_words
            ]

    lemmatized = [
    WordNetLemmatizer().lemmatize(word, pos = "v")
    for word in tokenized_sentence_cleaned
    ]

    cleaned_sentence = ' '.join(word for word in lemmatized)


    return cleaned_sentence


def X_y(df, TARGET_COLUMN):
    X = df.drop([TARGET_COLUMN], axis=1)
    y = df[TARGET_COLUMN]
    return X, y

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y,\
        test_size=0.3, random_state=0, shuffle=True)

    return X_train, X_test, y_train, y_test
