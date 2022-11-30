from ml_logic.preprocessor import *
from ml_logic.params import TARGET_COLUMN



def split_data(raw_data):

    selected_columns = preparation(raw_data)

    # cleaned_columns =  selected_columns.apply(clean)

    X, y = X_y(selected_columns, TARGET_COLUMN)

    X_train, X_test, y_train, y_test = split_data(X, y)

    # X_train_vect = train_vect(X_train)
    # X_train_trans = transform_vect(X_train_vect)

    # X_test_vect = train_vect(X_test)

    return X_train, X_test, y_train, y_test

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

cleaner = FunctionTransformer(clean)

preprocessor = Pipeline([
    ('cleaner', cleaner)
                         ])
