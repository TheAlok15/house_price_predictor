# src/model.py
from sklearn.linear_model import LinearRegression
import joblib

def create_model():
    model = LinearRegression()
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def predict_prices(model, X_test):
    predictions = model.predict(X_test)
    return predictions

def save_model(model, filepath):
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model

if __name__ == '__main__':
    linear_reg_model = create_model()
    print("Linear Regression model created:", linear_reg_model)

    # Note: We won't be able to fully test training and prediction here without loading actual data. 
    # These functionalities will be used in the train.py and predict.py scripts.