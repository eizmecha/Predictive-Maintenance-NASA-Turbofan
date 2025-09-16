"""
Project: Predictive-Maintenance-NASA-Turbofan
Phase: 5 - Model Optimization
Script: optimize_model.py
Author: Eiz
Team: Data Detectives

Description: This script performs hyperparameter tuning on the best baseline model (Random Forest)
from Phase 4 using RandomizedSearchCV. The goal is to find the optimal set of parameters
that minimizes the RMSE on the validation set.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import os

# Set random state for reproducibility
RANDOM_STATE = 42

def load_data(data_path):
    """
    Loads the processed training data.
    
    Args:
        data_path (str): Path to the processed CSV file.

    Returns:
        tuple: (X_train, X_val, y_train, y_val, feature_columns)
    """
    print("📂 Loading processed data...")
    # Check if the file exists before trying to load it
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Processed data file not found at: {data_path}\n"
                                "Please ensure you have run 'preprocessing.py' first and that the file path is correct.")
    
    df = pd.read_csv(data_path)
    
    # Separate features and target
    feature_columns = [col for col in df.columns if col not in ['unit_id', 'time_cycles', 'RUL']]
    X = df[feature_columns]
    y = df['RUL']
    
    # Use the same split as Phase 4 for a fair comparison
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    print(f"   Training set shape: {X_train.shape}")
    print(f"   Validation set shape: {X_val.shape}")
    return X_train, X_val, y_train, y_val, feature_columns

def run_randomized_search(X_train, y_train):
    """
    Executes the Randomized Search for Hyperparameter Tuning.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        RandomizedSearchCV: Fitted search object.
    """
    # Define the parameter distribution to sample from
    param_distributions = {
        'n_estimators': [100, 200, 300, 400, 500],      # Number of trees
        'max_depth': [None, 10, 20, 30, 40, 50],        # Maximum depth of trees
        'min_samples_split': [2, 5, 10],                # Minimum samples required to split a node
        'min_samples_leaf': [1, 2, 4],                  # Minimum samples required at a leaf node
        'max_features': ['auto', 'sqrt', 'log2'],       # Number of features to consider for splits
        'bootstrap': [True, False]                      # Whether to bootstrap samples
    }

    # Create the base model
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    # Setup the RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=50,  # Number of parameter combinations to try
        cv=5,       # 5-fold cross-validation
        scoring='neg_root_mean_squared_error', # We want to minimize RMSE
        verbose=3,  # High verbosity: show messages for each step
        random_state=RANDOM_STATE,
        n_jobs=-1   # Use all available CPU cores
    )

    print("🚀 Starting Randomized Search for Hyperparameter Tuning...")
    print("   This may take a while. Go grab a coffee!")
    print("="*60)
    
    start_time = time.time()
    # Fit the random search model
    random_search.fit(X_train, y_train)
    end_time = time.time()

    print("✅ Randomized Search Complete!")
    print(f"   Total tuning time: {(end_time - start_time) / 60:.2f} minutes")
    
    return random_search

def evaluate_optimized_model(best_model, X_val, y_val):
    """
    Evaluates the best model from the search on the validation set.

    Args:
        best_model: The best_estimator_ from RandomizedSearchCV.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
    """
    print("\n🧪 Evaluating Optimized Model on Validation Set...")
    y_pred = best_model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print("=== OPTIMIZED MODEL RESULTS ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print("="*30)

def main():
    """Main function to run the optimization pipeline."""
    print("🎯 PHASE 5: Random Forest Hyperparameter Optimization")
    print("==========================================================")
    
    # --- FIXED PATH CALCULATION --- 
    # Get the project root (two levels up from this script's directory: src/models/ -> src/ -> project_root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Define the CORRECT path to the processed data
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'train_FD001_processed.csv')
    model_save_path = os.path.join(project_root, 'models', 'optimized_random_forest_model.pkl')
    
    print(f"Project root is: {project_root}")
    print(f"Looking for data at: {processed_data_path}")
    # --- END FIX ---
    
    # 1. Load Data
    X_train, X_val, y_train, y_val, feature_columns = load_data(processed_data_path)
    
    # 2. Run Hyperparameter Tuning
    random_search = run_randomized_search(X_train, y_train)
    
    # 3. Print Best Parameters & Score
    print("\n🏆 BEST PARAMETERS FOUND")
    print("="*30)
    best_params = random_search.best_params_
    for key, value in best_params.items():
        print(f"{key}: {value}")
        
    best_score = -random_search.best_score_ # Remember score is negative RMSE
    print(f"\nBest Cross-Validation RMSE: {best_score:.4f}")
    
    # 4. Evaluate the best model on the hold-out validation set
    best_model = random_search.best_estimator_
    evaluate_optimized_model(best_model, X_val, y_val)
    
    # 5. Save the optimized model for future use
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(best_model, model_save_path)
    print(f"💾 Optimized model saved to: {model_save_path}")

if __name__ == "__main__":
    main()