from GEofnd.ml_logic.preprocessor import split_data,X_y, getdata, clean
from GEofnd.ml_logic.params import TARGET_COLUMN, FEATURE_COLUMN
from GEofnd.ml_logic.registry import save_model, load_model
from GEofnd.ml_logic.encoders import MNB, CountVect, pipe
from GEofnd.ml_logic.scraping_module import scraping
from GEofnd.ml_logic.classifier import result

import pandas as pd


#ok = 'https://www.cnbc.com/2022/12/05/russia-ukraine-live-updates.html'
#df_dataset = getdata()
url = {"okey": "https://www.cnbc.com/2022/12/05/russia-ukraine-live-updates.html"}

def prep_split_data(df=None):

    df['cleaned_news'] = df[FEATURE_COLUMN].apply(clean)
    X, y = X_y(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test


def train(X = None, y = None):

    X_train, X_test, y_train, y_test = prep_split_data(df_dataset)
    # model = load_model()
    # if model is None:
    model = pipe
    pipe_fitted = model.fit(X_train, y_train)
    save_model(model=pipe_fitted)

    return

def predict(url):


    url = list(url.items())[0][1]

    #if its a website
    if url.startswith('http') or url.startswith('www'):
        model = load_model()

        url_df = scraping(url)
        if url_df['news'][0] == None:
            return 'the bat seems tired today, please try by copy pasting the text of your article'

        #if the scraping does not work
        else :
            url_df['cleaned_news'] = url_df['news'].apply(clean)

            prediction = model.predict(url_df['cleaned_news'])
            predict_proba = model.predict_proba(url_df['cleaned_news'])

            label = result(prediction,predict_proba)
            return label

    #if it is a text
    else :

        model = load_model()

        url_df = pd.DataFrame({'news': [url]})
        url_df['cleaned_news'] = url_df['news'].apply(clean)

        prediction = model.predict(url_df['cleaned_news'])
        predict_proba = model.predict_proba(url_df['cleaned_news'])

        label = result(prediction,predict_proba)

        return label


if __name__ == '__main__':
#    prep_split_data(df_dataset)
#    train()
    predict(url)
