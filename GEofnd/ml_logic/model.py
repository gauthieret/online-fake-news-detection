import numpy as np
from GEofnd.ml_logic.encoders import CountVect, MNB
from GEofnd.ml_logic.encoders import MNB, CountVect
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer




def y_pred(MNB, tfidf_test):

    y_pred = MNB.predict(tfidf_test)

    return y_pred
