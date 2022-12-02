import pandas as pd
def getdata(data):
    data = f"../online-fake-news-detection/data/{data}.csv"
    data_df = pd.read_csv(data)
    return data_df
