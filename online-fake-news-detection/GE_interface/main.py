import numpy as np
import pandas as pd

from GE_ml_logic.data import data_df
from GE_ml_logic.scraping_module import get_new_data
from GE_ml_logic.model import train_vect, transform_vect, MNB_fit, y_pred
from GE_ml_logic.preprocessor import clean, X_y, split_data

# def new_data():
#     url = 'https://carnegieeurope.eu/2022/11/29/paradigm-shift-eu-russia-relations-after-war-in-ukraine-pub-88476'
#     return get_new_data(url)

# new_df = new_data()

# def preprocess_data():
#     clean_df = clean(new_df)
#     X_y_Data = X_y(clean_df)
#     splitted_data = split_data(X_y_Data)
#     preprocess_dataset()




# def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
#     y_pred(get_new_data)



# if __name__ == '__main__':
#     preprocess()
#     train()
#     pred()
#     evaluate()
