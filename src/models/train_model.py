"""
Model Training Module for Turbofan Engine RUL Prediction.

This module provides functions to train, evaluate, and analyze multiple machine learning models
for predicting the Remaining Useful Life (RUL) of aircraft engines.

Author: Osamah
Team: Data Detectives
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import os

# Set consistent plotting style
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def prepare_data(df, test_size=0.2, random_state=42):
    """
    Prepares the processed data for modeling by splitting into features and target,
    and then into training and validation sets.

    Args:
        df (pd.DataFrame): The fully processed dataframe from Phase 3.
        test_size (float): Proportion of data to use for validation.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: X_train, X_val, y_train, y_val, feature_columns
    """
    print("📊 Preparing data for modeling...")
    
    # Separate features (X) and target (y)
    # Exclude identifiers, time, and the target variable
    feature_columns = [col for col in df.columns if col not in ['unit_id', 'time_cycles', 'RUL']]
    X = df[feature_columns]
    y = df['RUL']
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Number of features: {len(feature_columns)}")
    
    return X_train, X_val, y_train, y_val, feature_columns


def evaluate_model(model, model_name, X_train, y_train, X_val, y_val):
    """
    Trains a model and evaluates its performance on the validation set.

    Args:
        model: sklearn-compatible model object
        model_name (str): Name of the model for reporting
        X_train, y_train: Training data
        X_val, y_val: Validation data

    Returns:
        tuple: trained_model, predictions, performance_metrics
    """
    print(f"\n🔍 Training {model_name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on validation set
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    # Create performance dictionary
    performance = {
        'RMSE': rmse,
        'MAE': mae, 
        'R2': r2,
        'Model': model
    }
    
    print(f"{model_name} Results:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}") 
    print(f"  R²: {r2:.4f}")
    
    return model, y_pred, performance


def train_all_models(X_train, y_train, X_val, y_val):
    """
    Trains and evaluates multiple regression models.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data

    Returns:
        dict: Dictionary containing performance metrics for all models
    """
    print("🚀 Training Multiple Models for Comparison")
    print("=" * 50)
    
    model_performance = {}
    
    # Model 1: Linear Regression
    lr_model, lr_preds, lr_perf = evaluate_model(
        LinearRegression(), "Linear Regression", X_train, y_train, X_val, y_val
    )
    model_performance['Linear Regression'] = lr_perf
    
    # Model 2: Random Forest
    rf_model, rf_preds, rf_perf = evaluate_model(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Random Forest", X_train, y_train, X_val, y_val
    )
    model_performance['Random Forest'] = rf_perf
    
    # Model 3: XGBoost
    xgb_model, xgb_preds, xgb_perf = evaluate_model(
        XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost", X_train, y_train, X_val, y_val
    )
    model_performance['XGBoost'] = xgb_perf
    
    return model_performance


def plot_performance_comparison(performance_dict):
    """
    Creates a visual comparison of model performance.

    Args:
        performance_dict (dict): Dictionary containing model performance metrics

    Returns:
        pd.DataFrame: DataFrame with performance metrics for all models
    """
    # Create DataFrame for easy comparison
    perf_df = pd.DataFrame.from_dict(
        {k: {m: v for m, v in v.items() if m != 'Model'} 
         for k, v in performance_dict.items()}, 
        orient='index'
    )
    
    # Sort by RMSE (lower is better)
    perf_df = perf_df.sort_values('RMSE')
    
    # Plot performance comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # RMSE Comparison
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax1.bar(perf_df.index, perf_df['RMSE'], color=colors, alpha=0.8)
    ax1.set_title('Model Comparison: RMSE (Lower is Better)')
    ax1.set_ylabel('Root Mean Squared Error')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom')
    
    # R² Comparison
    bars = ax2.bar(perf_df.index, perf_df['R2'], color=colors, alpha=0.8)
    ax2.set_title('Model Comparison: R² Score (Higher is Better)')
    ax2.set_ylabel('R² Score')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return perf_df


def plot_true_vs_predicted(y_true, y_pred, model_name):
    """
    Plots true vs predicted values for a model.

    Args:
        y_true: Actual target values
        y_pred: Predicted target values
        model_name (str): Name of the model for the plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, color='steelblue')
    
    # Add perfect prediction line
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('True RUL')
    plt.ylabel('Predicted RUL')
    plt.title(f'True vs Predicted RUL: {model_name}\n(RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_feature_importance(model, feature_names, model_name, top_n=15):
    """
    Plots feature importance for tree-based models.

    Args:
        model: Trained tree-based model (RandomForest or XGBoost)
        feature_names (list): List of feature names
        model_name (str): Name of the model for the plot title
        top_n (int): Number of top features to display
    """
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'get_booster'):
        importance = model.get_booster().get_score(importance_type='weight')
        # Convert to array format matching feature_names
        importance = np.array([importance.get(f, 0) for f in feature_names])
    else:
        print("Model doesn't have feature importance attribute")
        return
    
    # Create DataFrame
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(feat_imp)), feat_imp['importance'], color='#2E86AB')
    plt.yticks(range(len(feat_imp)), feat_imp['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances: {model_name}')
    plt.gca().invert_yaxis()  # Most important at top
    
    # Add value labels
    for i, (idx, row) in enumerate(feat_imp.iterrows()):
        plt.text(row['importance'] + 0.001, i, f'{row["importance"]:.3f}', 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return feat_imp


def save_model(model, filepath):
    """
    Saves a trained model to disk.

    Args:
        model: Trained model object
        filepath (str): Path where to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    joblib.dump(model, filepath)
    print(f"✅ Model saved to: {filepath}")


def main(data_path):
    """
    Main function to run the complete model training pipeline.

    Args:
        data_path (str): Path to the processed training data CSV file

    Returns:
        tuple: best_model, performance_dataframe
    """
    print("🎯 Starting Model Training Pipeline")
    print("=" * 50)
    
    # Check if data file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data file not found at: {data_path}")
    
    # Load processed data
    print("📁 Loading processed data...")
    processed_data = pd.read_csv(data_path)
    
    # Prepare data for modeling
    X_train, X_val, y_train, y_val, feature_columns = prepare_data(processed_data)
    
    # Train all models
    performance = train_all_models(X_train, y_train, X_val, y_val)
    
    # Compare performance
    perf_df = plot_performance_comparison(performance)
    
    # Identify best model
    best_model_name = perf_df.index[0]
    best_model = performance[best_model_name]['Model']
    best_preds = best_model.predict(X_val)
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Validation RMSE: {perf_df.loc[best_model_name, 'RMSE']:.2f}")
    
    # Plot true vs predicted for best model
    plot_true_vs_predicted(y_val, best_preds, best_model_name)
    
    # Plot feature importance for tree-based models
    if best_model_name in ['Random Forest', 'XGBoost']:
        feature_importance = plot_feature_importance(best_model, feature_columns, best_model_name)
    
    # Save the best model
    model_filename = f'../models/{best_model_name.lower().replace(" ", "_")}_model.pkl'
    save_model(best_model, model_filename)
    
    return best_model, perf_df

# ... (all the imports and function definitions remain the same until the main function)

def main(data_path=None):
    """
    Main function to run the complete model training pipeline.

    Args:
        data_path (str): Path to the processed training data CSV file. 
                         If None, uses default path from project root.

    Returns:
        tuple: best_model, performance_dataframe
    """
    print("🎯 Starting Model Training Pipeline")
    print("=" * 50)
    
    # Use default path if none provided
    if data_path is None:
        data_path = 'data/processed/train_FD001_processed.csv'
    
    # Check if data file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data file not found at: {data_path}\n"
                               f"Current working directory: {os.getcwd()}\n"
                               f"Please run the preprocessing phase first.")
    
    # Load processed data
    print("📁 Loading processed data...")
    processed_data = pd.read_csv(data_path)
    print(f"Loaded data shape: {processed_data.shape}")
    
    # Prepare data for modeling
    X_train, X_val, y_train, y_val, feature_columns = prepare_data(processed_data)
    
    # Train all models
    performance = train_all_models(X_train, y_train, X_val, y_val)
    
    # Compare performance
    perf_df = plot_performance_comparison(performance)
    
    # Identify best model
    best_model_name = perf_df.index[0]
    best_model = performance[best_model_name]['Model']
    best_preds = best_model.predict(X_val)
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Validation RMSE: {perf_df.loc[best_model_name, 'RMSE']:.2f}")
    
    # Plot true vs predicted for best model
    plot_true_vs_predicted(y_val, best_preds, best_model_name)
    
    # Plot feature importance for tree-based models
    if best_model_name in ['Random Forest', 'XGBoost']:
        feature_importance = plot_feature_importance(best_model, feature_columns, best_model_name)
    
    # Save the best model
    model_filename = f'models/{best_model_name.lower().replace(" ", "_")}_model.pkl'
    save_model(best_model, model_filename)
    
    return best_model, perf_df


# Only run if this script is executed directly (not when imported)
if __name__ == "__main__":
    # This will only run if you execute: python train_model.py
    try:
        main()  # No argument needed - uses default path
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Set consistent plotting style
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_processed_data(data_path):
    try:
        df = pd.read_csv(data_path)
        print(f"Processed data loaded successfully from: {data_path}")
        return df
    except FileNotFoundError:
        print(f" Error: Processed data file not found at {data_path}")
        print("Please run preprocessing first to create the processed data.")
        return None
    except Exception as e:
        print(f" Error loading processed data: {e}")
        return None

def prepare_data(df, test_size=0.2, random_state=42):
    print(" Preparing data for modeling...")
    
    # Separate features (X) and target (y)
    feature_columns = [col for col in df.columns if col not in ['unit_id', 'time_cycles', 'RUL']]
    X = df[feature_columns]
    y = df['RUL']
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Number of features: {len(feature_columns)}")
    
    return X_train, X_val, y_train, y_val, feature_columns

def evaluate_model(model, model_name, X_train, y_train, X_val, y_val):
    print(f"\n Training {model_name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on validation set
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    # Create performance dictionary
    performance = {
        'RMSE': rmse,
        'MAE': mae, 
        'R2': r2,
        'Model': model
    }
    
    print(f"{model_name} Results:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}") 
    print(f"  R²: {r2:.4f}")
    
    return model, y_pred, performance

def train_all_models(X_train, y_train, X_val, y_val):
    print(" Training Multiple Models for Comparison")
    print("=" * 50)
    
    model_performance = {}
    
    # Model 1: Linear Regression
    lr_model, lr_preds, lr_perf = evaluate_model(
        LinearRegression(), "Linear Regression", X_train, y_train, X_val, y_val
    )
    model_performance['Linear Regression'] = lr_perf
    
    # Model 2: Random Forest
    rf_model, rf_preds, rf_perf = evaluate_model(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Random Forest", X_train, y_train, X_val, y_val
    )
    model_performance['Random Forest'] = rf_perf
    
    # Model 3: XGBoost
    xgb_model, xgb_preds, xgb_perf = evaluate_model(
        XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost", X_train, y_train, X_val, y_val
    )
    model_performance['XGBoost'] = xgb_perf
    
    return model_performance

def plot_performance_comparison(performance_dict):
    # Create DataFrame for easy comparison
    perf_df = pd.DataFrame.from_dict(
        {k: {m: v for m, v in v.items() if m != 'Model'} 
         for k, v in performance_dict.items()}, 
        orient='index'
    )
    
    # Sort by RMSE (lower is better)
    perf_df = perf_df.sort_values('RMSE')
    
    # Plot performance comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # RMSE Comparison
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax1.bar(perf_df.index, perf_df['RMSE'], color=colors, alpha=0.8)
    ax1.set_title('Model Comparison: RMSE (Lower is Better)')
    ax1.set_ylabel('Root Mean Squared Error')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom')
    
    # R² Comparison
    bars = ax2.bar(perf_df.index, perf_df['R2'], color=colors, alpha=0.8)
    ax2.set_title('Model Comparison: R² Score (Higher is Better)')
    ax2.set_ylabel('R² Score')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return perf_df

def plot_true_vs_predicted(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, color='steelblue')
    
    # Add perfect prediction line
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('True RUL')
    plt.ylabel('Predicted RUL')
    plt.title(f'True vs Predicted RUL: {model_name}\n(RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_feature_importance(model, feature_names, model_name, top_n=15):
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'get_booster'):
        importance = model.get_booster().get_score(importance_type='weight')
        # Convert to array format matching feature_names
        importance = np.array([importance.get(f, 0) for f in feature_names])
    else:
        print("Model doesn't have feature importance attribute")
        return None
    
    # Create DataFrame
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(feat_imp)), feat_imp['importance'], color='#2E86AB')
    plt.yticks(range(len(feat_imp)), feat_imp['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances: {model_name}')
    plt.gca().invert_yaxis()  # Most important at top
    
    # Add value labels
    for i, (idx, row) in enumerate(feat_imp.iterrows()):
        plt.text(row['importance'] + 0.001, i, f'{row["importance"]:.3f}', 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return feat_imp

def save_model(model, filepath):
    # Create directory if it doesn't exist
    model_dir = os.path.dirname(filepath)
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, filepath)
    print(f" Model saved to: {filepath}")

def main(data_path=None):
    print(" Starting Model Training Pipeline")
    print("=" * 50)
    
    # Get the absolute path to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Use default path if not provided
    if data_path is None:
        data_path = os.path.join('data', 'processed', 'train_FD001_processed.csv')
    
    print(f"Looking for processed data at: {data_path}")
    
    # Load processed data
    processed_data = load_processed_data(data_path)
    if processed_data is None:
        exit(1)
    
    # Prepare data for modeling
    X_train, X_val, y_train, y_val, feature_columns = prepare_data(processed_data)
    
    # Train all models
    performance = train_all_models(X_train, y_train, X_val, y_val)
    
    # Compare performance
    perf_df = plot_performance_comparison(performance)
    
    # Identify best model
    best_model_name = perf_df.index[0]
    best_model = performance[best_model_name]['Model']
    best_preds = best_model.predict(X_val)
    
    print(f"\n Best Model: {best_model_name}")
    print(f"   Validation RMSE: {perf_df.loc[best_model_name, 'RMSE']:.2f}")
    
    # Plot true vs predicted for best model
    plot_true_vs_predicted(y_val, best_preds, best_model_name)
    
    # Plot feature importance for tree-based models
    if best_model_name in ['Random Forest', 'XGBoost']:
        feature_importance = plot_feature_importance(best_model, feature_columns, best_model_name)
    
    # Save the best model
    models_dir = os.path.join(project_root, 'models')
    model_filename = f'{best_model_name.lower().replace(" ", "_")}_model.pkl'
    model_path = os.path.join(models_dir, model_filename)
    save_model(best_model, model_path)
    
    return best_model, perf_df

if __name__ == "__main__":
    # Run the training pipeline
    main()