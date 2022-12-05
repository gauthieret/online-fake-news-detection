from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import unidecode
import nltk
import pandas as pd
from GEofnd.ml_logic.encoders import MNB, CountVect
import numpy as np
from GEofnd.ml_logic.params import TARGET_COLUMN, FEATURE_COLUMN, DATASET_LOCAL_PATH


def getdata():
    #articles = "/Users/Gauthier/code/gauthieret/online-fake-news-detection/GE_ofnd/data/DS2_fakenews.csv"
    #articles = "/Users/Gauthier/code/gauthieret/online-fake-news-detection/GE_ofnd/data/articles.csv"
    articles_df = pd.read_csv(DATASET_LOCAL_PATH)

    return articles_df


def clean(sentence):

    # Basic cleaning
    sentence = sentence.strip() ## remove whitespaces
    sentence = sentence.lower() ## lowercase
    sentence = ''.join(char for char in sentence if not char.isdigit()) ## remove numbers

    # Advanced cleaning
    for punctuation in (string.punctuation + "…”"):
        sentence = sentence.replace(punctuation, ' ') ## remove punctuation

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


def clean_data(X):
    return X['news'].apply(clean)


def X_y(df):
    X = df[FEATURE_COLUMN]
    y = df[TARGET_COLUMN]
    return X, y



def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y,\
        test_size=0.3, random_state=0, shuffle=True)

    return X_train, X_test, y_train, y_test



def fit_vect(X_train,y_train):

    CountVect_fit = CountVect.fit(X_train,y_train)

    return CountVect_fit



def transform_vect(X_train:pd.DataFrame, X_test:pd.DataFrame, y_train : pd.DataFrame):

    X_tain_trans = fit_vect(X_train,y_train).transform(X_train)
    X_test_trans = fit_vect(X_train,y_train).transform(X_test)

    return X_tain_trans, X_test_trans
