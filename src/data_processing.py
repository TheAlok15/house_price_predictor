import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    df = pd.get_dummies(
        df, 
        columns=['location'], prefix='location', drop_first=True
    )
    X = df.drop('price', axis=1)
    y = df['price']
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    data_path = "C:\\Users\\Alok singh\\projects\\housePricePrediction\\data\\house_price.csv"
    df = load_data(data_path)
    if df is not None:
        X, y = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_data(X, y)
        print("Shape of X_train:", X_train.shape)
        print("Shape of X_test:", X_test.shape)
        print("Shape of y_train:", y_train.shape)
        print("Shape of y_test:", y_test.shape)