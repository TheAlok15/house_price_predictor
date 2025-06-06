import os
import pandas as pd
from src.model import load_model, predict_prices
import joblib  # Import joblib

# Define paths
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'linear_regression_model.joblib')
FEATURES_PATH = os.path.join(MODEL_DIR, 'model_features.joblib')  # Path to saved features


def predict(data):
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