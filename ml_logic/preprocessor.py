from sklearn.model_selection import train_test_split
def X_y(df, target):
    X = df.drop([target], axis=1)
    y = df[target]
    return X, y

def split_data(tuple):

    X_train, X_test, y_train, y_test = train_test_split(tuple[0], tuple[1],\
        test_size=0.3, random_state=0, shuffle=True)

    return X_train, X_test, y_train, y_test

def preprocess(X_train, X_test):

    return X_train_preproc, X_test_preproc
