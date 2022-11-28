from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def train_model(X: np.ndarray):

    tfidf_vectorizer=TfidfVectorizer(max_df=0.7)
    tfidf=tfidf_vectorizer.fit(X)
    return tfidf

def transform_model(X: np.ndarray):

    tfidf_vectorizer=TfidfVectorizer(max_df=0.7)
    tfidf=tfidf_vectorizer.transform(X)

    return tfidf

def pa_classifier_fit(tfidf_train, y_train):

    pac=PassiveAggressiveClassifier(max_iter=50, n_jobs = -1, random_state=0, fit_intercept=False, early_stopping=True,
                                validation_fraction=0.2, n_iter_no_change=5)
    pac.fit(tfidf_train,y_train)

    return pac

def y_pred(pac, tfidf_test):

    y_pred=pac.predict(tfidf_test)

    return y_pred
