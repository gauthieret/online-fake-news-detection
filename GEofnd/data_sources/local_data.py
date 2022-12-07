import pandas as pd
def getdata():

    DS2_csv = "../Gofnd/data/DS2_fakenews.csv"
    DS2_df = pd.read_csv(DS2_csv)


    DS3_submit_csv = "../ofnd/data/DS3_submit.csv"
    DS3_submit_df = pd.read_csv(DS3_submit_csv)
    DS3_test_csv = "../ofnd/data/DS3_test.csv"
    DS3_test_df = pd.read_csv(DS3_test_csv)
    DS3_submit_test_df = pd.merge(DS3_submit_df,DS3_test_df,'inner')
    DS3_train_csv = "../ofnd/data/DS3_train.csv"
    DS3_train_df = pd.read_csv(DS3_train_csv)
    DS3_df = pd.merge(DS3_submit_test_df,DS3_train_df, 'outer')
    DS3_df['news'] = DS3_df['title'] + DS3_df['text']
    DS3_df = DS3_df.drop(columns = ['id','author','title','text'])
    DS3_df['label'] = DS3_df['label'].apply(lambda x: True if x == 1 else False)

    articles_df = pd.merge(DS2_df,DS3_df, 'outer')

    return articles_df
