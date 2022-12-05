from GEofnd.ml_logic.preprocessor import split_data,X_y, getdata, clean
from GEofnd.ml_logic.params import TARGET_COLUMN, FEATURE_COLUMN
from GEofnd.ml_logic.registry import save_model, load_model
from GEofnd.ml_logic.encoders import MNB, CountVect, pipe
from GEofnd.ml_logic.scraping_module import scraping
import pandas as pd


df_dataset = getdata()
url = 'https://edition.cnn.com/europe/live-news/russia-ukraine-war-news-11-29-22/index.html'
#url = 'https://www.bloomberg.com/news/articles/2022-12-01/kim-kardashian-s-investment-firm-hires-brisske-from-permira'

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

    if url.startswith('http') or url.startswith('www'):
        model = load_model()

        url_df = scraping(url)
        url_df['cleaned_news'] = url_df['news'].apply(clean)



        prediction = model.predict(url_df['cleaned_news'])
        predict_proba = model.predict_proba(url_df['cleaned_news'])

        if prediction[0] == True:
            if predict_proba[0][1] > 0.70:
                print("It's true!")
            elif predict_proba[0][1] > 0.5:
                print('Probably true')
            elif predict_proba[0][1] <= 0.5:
                print('possibly true dude')

        elif prediction[0] == False:
            if predict_proba[0][0] > 0.70:
                print("It's a fake news")
            elif predict_proba[0][0] > 0.5:
                print('Probably fake')
            elif predict_proba[0][1] <= 0.5:
                print('Possibly fake')

    else :

        model = load_model()
        url_df = pd.DataFrame({'news': [url]})

        url_df['cleaned_news'] = url_df['news'].apply(clean)



        prediction = model.predict(url_df['cleaned_news'])
        predict_proba = model.predict_proba(url_df['cleaned_news'])

        if prediction[0] == True:
            if predict_proba[0][1] > 0.70:
                print("It's true!")
            elif predict_proba[0][1] > 0.5:
                print('Probably true')
            elif predict_proba[0][1] <= 0.5:
                print('possibly true dude')

        elif prediction[0] == False:
            if predict_proba[0][0] > 0.70:
                print("It's a fake news")
            elif predict_proba[0][0] > 0.5:
                print('Probably fake')
            elif predict_proba[0][1] <= 0.5:
                print('Possibly fake')


if __name__ == '__main__':
#    prep_split_data(df_dataset)
#    train()
    predict(url)
