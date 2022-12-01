import pandas as pd

def get_data():
    new_csv = "../online-fake-news-detection/data/fakenews.csv"
    news_df = pd.read_csv(new_csv)
    return news_df

data_df = get_data()
data_df
