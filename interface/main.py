from ml_logic.preprocessor import *
from ml_logic.encoders import *
from ml_logic.params import TARGET_COLUMN



def split_data(raw_data):

    selected_columns = preparation(raw_data)

    X, y = X_y(selected_columns, TARGET_COLUMN)

    X_train, X_test, y_train, y_test = split_data(X, y)

    return X_train, X_test, y_train, y_test

def train(X, y):

    pipeline.fit(X, y)

    return pipeline

def predict(X_test):

    pipeline.predict(X_train, y_train)

    return model_trained

def predict(new_data):

     = pipe.predict(new_data)
