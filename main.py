"""
main.py - Main entry point for the C-MAPSS RUL Prediction Project.
Handles training, evaluation, and prediction using the saved models.
"""

import argparse
import os
import sys
import warnings
import pickle
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd

# Import from your custom src package
try:
    from data.preprocessing import load_raw_data, preprocess_data
    from models.train_model import train_all_models, prepare_data
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure your src package structure is correct and all modules exist.")
    sys.exit(1)

# Define base paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='C-MAPSS RUL Prediction Pipeline')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate a saved model on the test set')
    parser.add_argument('--predict', action='store_true', help='Make a prediction on new data')
    parser.add_argument('--model', type=str, choices=['random_forest', 'all', 'lstm'], default='all',
                        help='Model to use for training (all trains all models and selects best)')
    parser.add_argument('--data_path', type=str, help='Path to new data for prediction (CSV format)')
    parser.add_argument('--sequence_length', type=int, default=50, help='Sequence length for LSTM')
    return parser.parse_args()

def ensure_directories_exist():
    """Ensure all necessary directories exist."""
    for directory in [MODELS_DIR, RESULTS_DIR, PROCESSED_DATA_DIR]:
        directory.mkdir(exist_ok=True)

def train_random_forest(df):
    """
    Train a Random Forest model specifically.
    This function extracts the Random Forest model from the trained models.
    """
    print("Training Random Forest model specifically...")
    
    # Prepare data
    X_train, X_val, y_train, y_val, feature_columns = prepare_data(df)
    
    # Train all models and get the Random Forest one
    performance = train_all_models(X_train, y_train, X_val, y_val)
    
    # Return the Random Forest model
    return performance['Random Forest']['Model']

def prepare_sequences(df, sequence_length=50, target_column='RUL'):
    """
    Prepare sequences for LSTM training.
    This function creates sliding windows of sequences from the time series data.
    """
    sequences = []
    targets = []
    engine_ids = []
    
    # Group by engine_id to process each engine separately
    for engine_id, engine_data in df.groupby('unit_id'):
        engine_data = engine_data.sort_values('time_cycles')
        
        # Extract features (exclude non-feature columns)
        feature_columns = [col for col in engine_data.columns if col not in ['unit_id', 'time_cycles', target_column]]
        features = engine_data[feature_columns].values
        
        # Create sequences
        for i in range(len(engine_data) - sequence_length):
            sequence = features[i:i + sequence_length]
            target = engine_data[target_column].iloc[i + sequence_length]
            
            sequences.append(sequence)
            targets.append(target)
            engine_ids.append(engine_id)
    
    return np.array(sequences), np.array(targets), np.array(engine_ids)

def train_lstm(df, sequence_length=50):
    """
    Train an LSTM model (placeholder function - you would implement this based on your notebooks)
    """
    print("LSTM training is not implemented in this main pipeline.")
    print("Please use your Jupyter notebooks for LSTM training:")
    print(" - 03_train_models.ipynb for basic LSTM training")
    print(" - 05_optimize_lstm.ipynb for optimized LSTM training")
    
    # This is a placeholder - you would implement actual LSTM training here
    # based on your notebook code
    return None

def predict_on_test_data(model, scaler, test_file_path):
    """
    Make predictions on the test data and save results.
    """
    print(f"Loading test data from: {test_file_path}")
    
    # Load test data
    test_df = load_raw_data(test_file_path)
    if test_df is None:
        print("Failed to load test data. Exiting.")
        return None
    
    print(f"Test data shape: {test_df.shape}")
    
    # Load true RUL values for evaluation
    rul_file = DATA_DIR / 'RUL_FD001.txt'
    if rul_file.exists():
        true_rul = pd.read_csv(rul_file, header=None, names=['RUL']).squeeze()
    else:
        true_rul = None
        print("Warning: True RUL file not found. Only generating predictions.")
    
    # Preprocess test data using the already-fitted scaler
    test_df_processed, _ = preprocess_data(test_df, scaler=scaler, fit_scaler=False)
    
    # Group by engine and get the last data point for each engine
    last_points = test_df_processed.groupby('unit_id').last()
    X_test = last_points.drop(['RUL', 'time_cycles', 'unit_id'], axis=1, errors='ignore')
    
    print(f"Making predictions for {len(X_test)} engines...")
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'unit_id': last_points.index,
        'final_time_cycle': last_points['time_cycles'],
        'predicted_rul': predictions
    })
    
    # Add true RUL if available
    if true_rul is not None and len(true_rul) == len(results):
        results['true_rul'] = true_rul.values
        results['error'] = results['true_rul'] - results['predicted_rul']
        results['absolute_error'] = np.abs(results['error'])
        
        # Calculate overall metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(results['true_rul'], results['predicted_rul'])
        rmse = np.sqrt(mean_squared_error(results['true_rul'], results['predicted_rul']))
        r2 = r2_score(results['true_rul'], results['predicted_rul'])
        
        print(f"\n📊 Test Set Performance Metrics:")
        print(f"MAE: {mae:.2f} cycles")
        print(f"RMSE: {rmse:.2f} cycles")
        print(f"R² Score: {r2:.4f}")
        
        # Add performance metrics to results file
        metrics_df = pd.DataFrame({
            'metric': ['MAE', 'RMSE', 'R2'],
            'value': [mae, rmse, r2]
        })
        metrics_path = RESULTS_DIR / 'test_performance_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Performance metrics saved to {metrics_path}")
    
    return results

def main():
    args = parse_args()
    ensure_directories_exist()
    
    if args.train:
        print(f"Training {args.model} model...")
        
        # Load and preprocess training data
        print("1. Loading and preprocessing data...")
        train_file = DATA_DIR / 'train_FD001.txt'
        train_df = load_raw_data(train_file)
        
        if train_df is None:
            print("Failed to load training data. Exiting.")
            return
        
        train_df_processed, scaler = preprocess_data(train_df, fit_scaler=True)
        
        # Save the fitted scaler (important for consistency later)
        scaler_path = MODELS_DIR / 'fitted_scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Fitted scaler saved to {scaler_path}")
        
        # Save processed data for reference
        processed_data_path = PROCESSED_DATA_DIR / 'train_FD001_processed.csv'
        train_df_processed.to_csv(processed_data_path, index=False)
        print(f"Processed training data saved to {processed_data_path}")
        
        # Train the selected model
        if args.model == 'random_forest':
            print("2. Training Random Forest model...")
            model = train_random_forest(train_df_processed)
            model_path = MODELS_DIR / 'random_forest_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Random Forest model saved to {model_path}")
            print("Training completed successfully!")
            
        elif args.model == 'all':
            print("2. Training all models and selecting the best one...")
            # Prepare data
            X_train, X_val, y_train, y_val, feature_columns = prepare_data(train_df_processed)
            
            # Train all models
            performance = train_all_models(X_train, y_train, X_val, y_val)
            
            # Find the best model (lowest RMSE)
            best_model_name = min(performance.keys(), key=lambda x: performance[x]['RMSE'])
            best_model = performance[best_model_name]['Model']
            
            # Save the best model
            model_filename = f'{best_model_name.lower().replace(" ", "_")}_model.pkl'
            model_path = MODELS_DIR / model_filename
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            print(f"Best model ({best_model_name}) saved to {model_path}")
            
        elif args.model == 'lstm':
            print("2. Training LSTM model...")
            model = train_lstm(train_df_processed, args.sequence_length)
            if model is not None:
                model_path = MODELS_DIR / 'lstm_model.keras'
                model.save(model_path)
                print(f"LSTM model saved to {model_path}")
    
    elif args.evaluate:
        print(f"Evaluating model on test set...")
        
        # Load test data and true RUL values
        test_file = DATA_DIR / 'test_FD001.txt'
        rul_file = DATA_DIR / 'RUL_FD001.txt'
        
        test_df = load_raw_data(test_file)
        if test_df is None:
            print("Failed to load test data. Exiting.")
            return
            
        true_rul = pd.read_csv(rul_file, header=None, names=['RUL']).squeeze()
        
        # Load the fitted scaler (do not fit a new one!)
        scaler_path = MODELS_DIR / 'fitted_scaler.pkl'
        scaler = joblib.load(scaler_path)
        
        # Preprocess test data using the already-fitted scaler
        test_df_processed, _ = preprocess_data(test_df, scaler=scaler, fit_scaler=False)
        
        # Try to load the Random Forest model
        model_path = MODELS_DIR / 'random_forest_model.pkl'
        if not model_path.exists():
            print(f"Model file not found at: {model_path}")
            print("Please train a model first.")
            return
            
        model = joblib.load(model_path)
        
        # For evaluation, use the last data point of each engine
        last_points = test_df_processed.groupby('unit_id').last()
        X_test = last_points.drop(['RUL', 'time_cycles', 'unit_id'], axis=1, errors='ignore')
        predictions = model.predict(X_test)
        
        results = pd.DataFrame({
            'unit_id': last_points.index,
            'true_rul': true_rul.values,
            'predicted_rul': predictions
        })
        
        # Calculate evaluation metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(results['true_rul'], results['predicted_rul'])
        rmse = np.sqrt(mean_squared_error(results['true_rul'], results['predicted_rul']))
        r2 = r2_score(results['true_rul'], results['predicted_rul'])
        
        print(f"Evaluation Metrics:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.4f}")
        
        # Save results
        results_path = RESULTS_DIR / 'final_test_results.csv'
        results.to_csv(results_path, index=False)
        print(f"Evaluation results saved to {results_path}")
    
    elif args.predict:
        print(f"Making predictions using model...")
        
        # Load the fitted scaler and model using joblib
        scaler_path = MODELS_DIR / 'fitted_scaler.pkl'
        if not scaler_path.exists():
            print("Fitted scaler not found. Please train a model first.")
            return
            
        scaler = joblib.load(scaler_path)
        
        # Try to load the Random Forest model
        model_path = MODELS_DIR / 'random_forest_model.pkl'
        if not model_path.exists():
            print("Random Forest model not found. Please train a model first.")
            return
            
        model = joblib.load(model_path)
        
        # Determine which data file to use
        if args.data_path:
            # Use the specified data file
            test_file_path = Path(args.data_path)
            if not test_file_path.exists():
                print(f"Error: File '{args.data_path}' does not exist.")
                print("Using default test data: data/test_FD001.txt")
                test_file_path = DATA_DIR / 'test_FD001.txt'
        else:
            # Use the default test data
            test_file_path = DATA_DIR / 'test_FD001.txt'
        
        # Make predictions on test data
        results = predict_on_test_data(model, scaler, test_file_path)
        
        if results is not None:
            # Save predictions
            predictions_path = RESULTS_DIR / 'test_predictions.csv'
            results.to_csv(predictions_path, index=False)
            print(f"Predictions saved to {predictions_path}")
            
            # Show some sample predictions
            print("\nSample predictions:")
            print(results.head(10))
            
            # Basic statistics
            print(f"\nPrediction Statistics:")
            print(f"Number of engines predicted: {len(results)}")
            print(f"Average predicted RUL: {results['predicted_rul'].mean():.2f}")
            print(f"Minimum predicted RUL: {results['predicted_rul'].min():.2f}")
            print(f"Maximum predicted RUL: {results['predicted_rul'].max():.2f}")
            
            # Calculate prediction confidence intervals
            std_dev = results['predicted_rul'].std()
            print(f"Standard Deviation: {std_dev:.2f}")
            print(f"95% Confidence Interval: ({results['predicted_rul'].mean() - 1.96*std_dev:.2f}, "
                  f"{results['predicted_rul'].mean() + 1.96*std_dev:.2f})")
            
            # Identify engines that might need immediate attention
            critical_engines = results[results['predicted_rul'] < 30]
            if len(critical_engines) > 0:
                print(f"\n🚨 {len(critical_engines)} engine(s) may need immediate attention (RUL < 30):")
                for _, engine in critical_engines.iterrows():
                    print(f"   Engine {engine['unit_id']}: {engine['predicted_rul']:.1f} cycles remaining")
            else:
                print(f"\n✅ No engines require immediate attention (all RUL > 30 cycles)")
    
    else:
        print("Please specify an action: --train, --evaluate, or --predict")
        print("Example: python main.py --train --model random_forest")
        print("Example: python main.py --train --model all (to train all models and select best)")
        print("Example: python main.py --predict --model random_forest (uses test_FD001.txt)")
        print("Example: python main.py --predict --model random_forest --data_path path/to/your_data.txt")
        print("Example: python main.py --evaluate")

if __name__ == "__main__":
    main()