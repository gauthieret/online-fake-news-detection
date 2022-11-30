from ml_logic.preprocessor import *
from ml_logic.encoders import *
from ml_logic.params import TARGET_COLUMN
from ml_logic.registry import *
from ml_logic.model import *


def split_data(raw_data):

    selected_columns = preparation(raw_data)

    X, y = X_y(selected_columns, TARGET_COLUMN)

    X_train, X_test, y_train, y_test = split_data(X, y)

    return X_train, X_test, y_train, y_test

def train(X, y):
    model = load_model()
    if model is None:
        model = initialize_model()

    model.fit(X, y)

    save_model(model=model)

    return

def predict(X_test):

    y_pred = pipeline.predict(X_test)

    return y_pred
