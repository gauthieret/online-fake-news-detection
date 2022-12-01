from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from ml_logic.preprocessor import *

tfidf_vectorizer = TfidfVectorizer(max_df=0.7, min_df=0.2, max_features=5000)

ohe = OneHotEncoder(dtype=int, sparse=False)

cleaner = FunctionTransformer(clean_data)

pac = PassiveAggressiveClassifier(max_iter=50, n_jobs=-1, random_state=0, fit_intercept=False, early_stopping=True,
                                validation_fraction=0.2, n_iter_no_change=5)

topic_pipe = Pipeline([('ohe', ohe)])

text_pipe = Pipeline([
    ('cleaner', cleaner),
    ('tfidf_vectorizer', tfidf_vectorizer)
    ])

preproc_pipe = ColumnTransformer([
    ('text_pipe', text_pipe, 'news')
])

pipeline = make_pipeline(preproc_pipe, pac)
