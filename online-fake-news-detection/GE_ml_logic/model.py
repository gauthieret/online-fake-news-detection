from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def train_vect(X: np.ndarray):
    count_vec = CountVectorizer(binary=False, decode_error='strict', encoding='utf-8',
                            input='content', lowercase=True, max_df=1.0,
                            max_features=None, min_df=1,
                            ngram_range=(1, 1), preprocessor=None,
                            stop_words=None, strip_accents=None,
                            token_pattern='(?u)\\b\\w\\w+\\b',
                            tokenizer=None, vocabulary=None)
    tfidf_fitted = count_vec.fit(X)

    return tfidf_fitted


def transform_vect(X: np.ndarray, tfdidf_fitted):
    tfidf_transformed = tfdidf_fitted.transform(X)

    return tfidf_transformed


def MNB_fit(tfidf_train, y_train):

    MNB = MultinomialNB(alpha=1.0,class_prior=None,fit_prior=True),

    MNB.fit(tfidf_train,y_train)

    return MNB


def y_pred(MNB, tfidf_test):

    y_pred = MNB.predict(tfidf_test)

    return y_pred
