import pandas as pd
import joblib
import os

def load_raw_data(file_path, column_names):

    return pd.read_csv(file_path, delim_whitespace=True, header=None, names=column_names)

def save_model(model, file_path):
    """Saves a trained model to a file using joblib."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def load_model(file_path):
    """Loads a saved model from a file."""
    return joblib.load(file_path)

def save_processed_data(df, file_path):
    """Saves a processed DataFrame to a CSV file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Processed data saved to {file_path}")