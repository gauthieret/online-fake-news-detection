from ml_logic.preprocessor import *
from ml_logic.encoders import *
from ml_logic.params import TARGET_COLUMN
from ml_logic.registry import *
from ml_logic.model import *


def prep_split_data(raw_data):

    X, y = X_y(raw_data, TARGET_COLUMN)

    X_train, X_test, y_train, y_test = split_data(X, y)

    return X_train, X_test, y_train, y_test

def train(X, y):
    model = load_model()
    if model is None:
        model = initialize_model()

    model.fit(X, y)

    save_model(model=model)

    return

def predict(X_pred):

    model = load_model()

    prediction = model.predict(X_pred)[0]
    score = pipeline.decision_function(X_pred)[0]

    prediction_score = prediction, score

    return prediction_score
