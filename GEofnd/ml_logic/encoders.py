from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.pipeline import make_pipeline, Pipeline


CountVect = CountVectorizer(binary=False, decode_error='strict',
                                    encoding='utf-8',
                                    input='content', lowercase=True, max_df=1.0,
                                    max_features=None, min_df=1,
                                    ngram_range=(1, 1), preprocessor=None,
                                    stop_words=None, strip_accents=None,
                                    token_pattern='(?u)\\b\\w\\w+\\b',
                                    tokenizer=None, vocabulary=None)


MNB = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)


pipe = make_pipeline(CountVect,MNB)
