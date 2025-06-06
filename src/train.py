import os
import pandas as pd # Import pandas
from src.data_processing import load_data, preprocess_data, split_data
from src.model import create_model, train_model, save_model
import joblib

DATA_PATH = 'data/house_price.csv'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'linear_regression_model.joblib')
FEATURES_PATH = os.path.join(MODEL_DIR, 'model_features.joblib')

# os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    if df is not None:
        print("Preprocessing data...")
        X, y = preprocess_data(df)

        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

        joblib.dump(X_train.columns.tolist(), FEATURES_PATH) # Save feature names
        print(f"Feature names saved to {FEATURES_PATH}")

        print("Creating the linear regression model...")
        model = create_model()

        print("Training the model...")
        trained_model = train_model(model, X_train, y_train)

        print(f"Model trained successfully!")

        # Save the trained model
        save_model(trained_model, MODEL_PATH)

if __name__ == "__main__":
    main()