# # src/predict.py
# import os
# import pandas as pd
# from src.model import load_model, predict_prices
# from src.data_processing import preprocess_data  # We need this for consistent feature processing

# # Define paths
# MODEL_DIR = 'models'
# MODEL_PATH = os.path.join(MODEL_DIR, 'linear_regression_model.joblib')

# def predict(data):
#     """
#     Loads the trained model and makes predictions on the input data.

#     Args:
#         data (pd.DataFrame): DataFrame containing the features for prediction.
#                            It should have the same columns as the training data
#                            (sqft, bedrooms, bathrooms, location).

#     Returns:
#         list: A list of predicted house prices.
#     """
#     try:
#         loaded_model = load_model(MODEL_PATH)
#         # Ensure the input data is preprocessed in the same way as the training data
#         # We might need to adjust this based on how 'preprocess_data' was designed.
#         # For simplicity here, we'll assume 'preprocess_data' can handle a single row
#         # or a DataFrame and output the processed features.
#         X_processed, _ = preprocess_data(data.copy()) # We only need the features
#         predictions = predict_prices(loaded_model, X_processed)
#         return predictions
#     except FileNotFoundError:
#         print(f"Error: Model file not found at {MODEL_PATH}. Please train the model first.")
#         return None
#     except Exception as e:
#         print(f"An error occurred during prediction: {e}")
#         return None

# if __name__ == "__main__":
#     # Example usage: Creating a sample DataFrame for prediction
#     new_data = pd.DataFrame({
#         'sqft': [1500, 1200],
#         'bedrooms': [3, 2],
#         'bathrooms': [2, 1.5],
#         'location': ['Delhi', 'Maharashtra']
#     })

#     predicted_prices = predict(new_data)

#     if predicted_prices is not None:
#         print("\nPredicted Prices:")
#         for i, price in enumerate(predicted_prices):
#             print(f"House {i+1}: ₹{price:,.2f}")

import os
import pandas as pd
from src.model import load_model, predict_prices
import joblib  # Import joblib

# Define paths
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'linear_regression_model.joblib')
FEATURES_PATH = os.path.join(MODEL_DIR, 'model_features.joblib')  # Path to saved features


def predict(data):
    """
    Loads the trained model and makes predictions on the input data.

    Args:
        data (pd.DataFrame): DataFrame containing the features for prediction.
                           It should have the same columns as the training data
                           (sqft, bedrooms, bathrooms, location).

    Returns:
        list: A list of predicted house prices.
    """
    try:
        loaded_model = load_model(MODEL_PATH)
        # Load the expected feature names
        expected_features = joblib.load(FEATURES_PATH)

        # Replicate the one-hot encoding step directly for the input data
        X_processed = pd.get_dummies(data.copy(), columns=['location'], prefix='location',
                                     drop_first=True)

        # --- CRITICAL STEP: Reindex to align columns ---
        X_processed = X_processed.reindex(columns=expected_features, fill_value=0)
        # --- END OF CRITICAL STEP ---

        predictions = predict_prices(loaded_model, X_processed)
        return predictions
    except FileNotFoundError:
        print(
            f"Error: Model file not found at {MODEL_PATH} or feature file at {FEATURES_PATH}. Please train the model first.")
        return None
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None


if __name__ == "__main__":
    # Example usage: Creating a sample DataFrame for prediction
    new_data = pd.DataFrame({
        'sqft': [1500, 1200, 1800, 2000],  # Added more data
        'bedrooms': [3, 2, 3, 4],
        'bathrooms': [2, 1.5, 2.5, 3.0],
        'location': ['Delhi', 'Maharashtra', 'Karnataka',
                     'Tamil Nadu']  # Added more locations
    })

    predicted_prices = predict(new_data)

    if predicted_prices is not None:
        print("\nPredicted Prices:")
        for i, price in enumerate(predicted_prices):
            print(f"House {i + 1}: ₹{price:,.2f}")