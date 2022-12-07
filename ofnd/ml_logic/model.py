from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from ofnd.ml_logic.encoders import *
from ofnd.ml_logic.params import *
from transformers import RobertaForSequenceClassification



# this function is probably not needed
def pa_classifier_fit(tfidf_train, y_train):

    pac=PassiveAggressiveClassifier(max_iter=50, n_jobs=-1, random_state=0, fit_intercept=False, early_stopping=True,
                                validation_fraction=0.2, n_iter_no_change=5)
    pac.fit(tfidf_train,y_train)

    return pac



def y_pred(pac, tfidf_test):

    y_pred = pac.predict(tfidf_test)

    return y_pred

# this function has been tested and returns a pipeline
def initialize_model():
    '''Function to restart a model when there is not model
    in LOCAL_REGISTRY_PATH'''
    if MODEL_TYPE == "ml":
        text_pipe = Pipeline([
            ('cleaner', cleaner),
            ('tfidf_vectorizer', tfidf_vectorizer)
        ])

        preproc_pipe = ColumnTransformer([
            ('text_pipe', text_pipe, 'title_text')
        ])

        pipeline = make_pipeline(preproc_pipe, pac)

        return pipeline

    if MODEL_TYPE == 'tensorflow':
        model = Sequential()
        model.add(layers.Conv1D(16, 3))
        model.add(layers.Flatten())
        model.add(layers.Dense(5,))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
