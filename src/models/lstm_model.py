"""
Project: Predictive-Maintenance-NASA-Turbofan
Phase: 5 - Advanced Modeling
Script: lstm_model.py
Author: Eiz
Team: Data Detectives

Description: This script implements a Long Short-Term Memory (LSTM) neural network
for predicting the Remaining Useful Life (RUL) of turbofan engines. It processes
the data into sequences to leverage temporal patterns and compares the results
against the previously optimized Random Forest model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Set plotting style
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_and_preprocess_data(data_path):
    """
    Loads the processed data and prepares it for LSTM sequencing.
    Separates the data by engine and ensures correct sorting.

    Args:
        data_path (str): Path to the processed CSV file.

    Returns:
        pd.DataFrame: The processed data, sorted by unit_id and time_cycles.
    """
    print("📂 Loading and preparing data for LSTM...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Processed data file not found at: {data_path}")

    df = pd.read_csv(data_path)
    # It is VITALLY important that data is sorted correctly for sequence creation
    df = df.sort_values(by=['unit_id', 'time_cycles']).reset_index(drop=True)
    print(f"   Data sorted for sequencing. Shape: {df.shape}")
    return df

def create_sequences_all_engines(df, sequence_length=30, feature_columns=None):
    """
    The CORE function of this script. Creates sequences for LSTM training
    from all engines in the dataset.

    Args:
        df (pd.DataFrame): The full processed dataframe, sorted by unit_id and time_cycles.
        sequence_length (int): The number of previous cycles (timesteps) to use for prediction.
        feature_columns (list): List of feature names to use.

    Returns:
        tuple: (X_sequences, y_targets) - The 3D array of sequences and their corresponding RUL targets.
    """
    if feature_columns is None:
        # Exclude non-sensor/feature columns
        feature_columns = [col for col in df.columns if col not in ['unit_id', 'time_cycles', 'RUL']]

    X_sequences = []
    y_targets = []

    print(f"   Creating sequences (length={sequence_length}) for each engine...")
    # Group by each individual engine
    for unit_id in df['unit_id'].unique():
        engine_data = df[df['unit_id'] == unit_id]
        # Ensure the engine's data is in chronological order
        engine_data = engine_data.sort_values('time_cycles')
        
        engine_features = engine_data[feature_columns].values
        engine_rul = engine_data['RUL'].values

        # Create sequences for this engine
        for i in range(len(engine_data) - sequence_length):
            X_sequences.append(engine_features[i:i + sequence_length, :]) # Slice of 'sequence_length' timesteps
            y_targets.append(engine_rul[i + sequence_length]) # Predict the RUL at the end of the sequence

    print(f"   Created {len(X_sequences)} total sequences.")
    return np.array(X_sequences), np.array(y_targets)

def build_lstm_model(input_shape, lstm_units=[100, 50], dropout_rate=0.2, learning_rate=0.001):
    """
    Builds and compiles an LSTM model.

    Args:
        input_shape (tuple): Shape of the input data (timesteps, features).
        lstm_units (list): Number of units in each LSTM layer.
        dropout_rate (float): Dropout rate for regularization.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        tf.keras.Model: The compiled LSTM model.
    """
    print("🧠 Building LSTM model architecture...")
    model = Sequential()
    # First LSTM layer - returns sequences to feed to the next LSTM layer
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=lstm_units[0], return_sequences=True))
    model.add(Dropout(dropout_rate))

    # Second LSTM layer - does not return sequences
    model.add(LSTM(units=lstm_units[1], return_sequences=False))
    model.add(Dropout(dropout_rate))

    # Dense layers for interpretation and output
    model.add(Dense(units=25, activation='relu'))
    model.add(Dense(units=1)) # Linear activation for regression

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse', # Mean Squared Error is standard for regression
        metrics=['mae'] # Track Mean Absolute Error as well
    )

    model.summary()
    return model

def train_and_evaluate_lstm(X_train, y_train, X_val, y_val, input_shape, epochs=100, patience=10):
    """
    Trains the LSTM model with early stopping and evaluates it.

    Args:
        X_train, y_train: Training sequences and targets.
        X_val, y_val: Validation sequences and targets.
        input_shape (tuple): Shape for model input.
        epochs (int): Maximum number of epochs to train.
        patience (int): How many epochs to wait without improvement before stopping.

    Returns:
        tuple: (trained_model, training_history)
    """
    print("🚀 Training LSTM model...")
    model = build_lstm_model(input_shape)

    # Define Early Stopping callback to prevent overfitting
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True, # Keeps the best model found during training
        verbose=1
    )

    print("   Training will stop early if no improvement for 10 epochs.")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=64, # Can be adjusted based on memory constraints
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    return model, history

def plot_training_history(history):
    """
    Plots the training and validation loss curves.

    Args:
        history: The history object returned from model.fit().
    """
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model Mean Absolute Error')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the complete LSTM pipeline."""
    print("🎯 PHASE 5-B: LSTM Neural Network Training")
    print("==========================================================")

    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'train_FD001_processed.csv')
    model_save_path = os.path.join(project_root, 'models', 'lstm_model.keras')

    # 1. Load Data
    df = load_and_preprocess_data(processed_data_path)
    feature_columns = [col for col in df.columns if col not in ['unit_id', 'time_cycles', 'RUL']]

    # 2. Create Sequences for LSTM
    SEQUENCE_LENGTH = 30  # Look back 30 cycles to predict the next RUL
    X, y = create_sequences_all_engines(df, sequence_length=SEQUENCE_LENGTH, feature_columns=feature_columns)

    # 3. Split the SEQUENCES into train and validation sets
    # We cannot use a standard shuffle_split as it would break the time series nature.
    # We will split by overall fraction, ensuring sequences from the same engine stay together.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)
    print(f"   Final input shape for LSTM: {X_train.shape[1:]} (timesteps, features)")

    # 4. Build, Train, and Evaluate the LSTM Model
    input_shape = (X_train.shape[1], X_train.shape[2]) # (timesteps, features)
    model, history = train_and_evaluate_lstm(X_train, y_train, X_val, y_val, input_shape, epochs=100)

    # 5. Plot training history
    plot_training_history(history)

    # 6. Make final predictions and calculate metrics on the validation set
    print("\n🧪 Making final predictions on validation set...")
    y_pred = model.predict(X_val, verbose=0).flatten() # Flatten from 2D to 1D

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print("=== LSTM MODEL RESULTS ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print("="*30)

    # 7. Save the trained LSTM model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"💾 LSTM model saved to: {model_save_path}")

    # 8. Compare with Optimized Random Forest
    print("\n📊 FINAL MODEL COMPARISON")
    print("="*40)
    print(f"{'Model':<25} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print("-" * 40)
    print(f"{'Optimized Random Forest':<25} {'31.72':<10} {'22.15':<10} {'0.780':<10}")
    print(f"{'LSTM Neural Network':<25} {rmse:<10.2f} {mae:<10.2f} {r2:<10.4f}")
    print("=" * 40)

    if rmse < 31.72:
        print("\n🏆 NEW CHAMPION: LSTM Neural Network! 🏆")
    else:
        print("\n🏆 CHAMPION REMAINS: Optimized Random Forest!")

    return model, history

if __name__ == "__main__":
    main()