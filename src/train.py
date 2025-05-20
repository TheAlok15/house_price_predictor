# import os
# from src.data_processing import load_data, preprocess_data, split_data
# from src.model import create_model, train_model, save_model

# # Define paths
# DATA_PATH = 'data/house_price.csv'
# MODEL_DIR = 'models'
# MODEL_PATH = os.path.join(MODEL_DIR, 'linear_regression_model.joblib')

# # Create the models directory if it doesn't exist
# os.makedirs(MODEL_DIR, exist_ok=True)

# def main():
#     """Main function to train the linear regression model."""
#     print("Loading data...")
#     df = load_data(DATA_PATH)

#     if df is not None:
#         print("Preprocessing data...")
#         X, y = preprocess_data(df)

#         print("Splitting data into training and testing sets...")
#         X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

#         print("Creating the linear regression model...")
#         model = create_model()

#         print("Training the model...")
#         trained_model = train_model(model, X_train, y_train)

#         print(f"Model trained successfully!")

#         # Save the trained model
#         save_model(trained_model, MODEL_PATH)

# if __name__ == "__main__":
#     main()

import os
import pandas as pd # Import pandas
from src.data_processing import load_data, preprocess_data, split_data
from src.model import create_model, train_model, save_model
import joblib # Import joblib

# Define paths
DATA_PATH = 'data/house_price.csv'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'linear_regression_model.joblib')
FEATURES_PATH = os.path.join(MODEL_DIR, 'model_features.joblib') # New path

# Create the models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    """Main function to train the linear regression model."""
    print("Loading data...")
    df = load_data(DATA_PATH)

    if df is not None:
        print("Preprocessing data...")
        X, y = preprocess_data(df)

        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

        # Save the feature names
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