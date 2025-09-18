import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set random seed for reproducibility
RANDOM_STATE = 42

def load_raw_test_data(file_path):
    """
    Loads the raw test data with the same column structure as training data.
    """
    
    column_names = ['unit_id', 'time_cycles'] + [f'op_setting_{i}' for i in range(1,4)] + [f'sensor_{i}' for i in range(1,22)]
    
    print(f"📂 Loading raw test data from: {file_path}")
    test_df = pd.read_csv(file_path, sep=r'\s+', header=None, names=column_names)
    print(f"   Raw test data shape: {test_df.shape}")
    return test_df

def load_true_rul_data(file_path):
    """
    Loads the true RUL values for the test set.
    """
    print(f"📂 Loading true RUL values from: {file_path}")
    true_rul_df = pd.read_csv(file_path, header=None, names=['True_RUL'])
    print(f"   True RUL data shape: {true_rul_df.shape}")  # FIXED: changed 'true_rul' to 'true_rul_df'
    return true_rul_df

def preprocess_test_data(raw_test_df, fitted_scaler, important_sensors):
    """
    Applies the EXACT same preprocessing pipeline to the test data
    that was used on the training data.
    """
    print("🔄 Applying preprocessing pipeline to test data...")
    
    # Step 1: Add RUL column (for consistency, though we won't use it for prediction)
    # For test data, we need to calculate max cycle per engine
    max_cycle_per_engine = raw_test_df.groupby('unit_id')['time_cycles'].max().reset_index()
    max_cycle_per_engine.rename(columns={'time_cycles': 'max_cycle'}, inplace=True)
    
    test_df_with_rul = raw_test_df.merge(max_cycle_per_engine, on='unit_id', how='left')
    test_df_with_rul['RUL'] = test_df_with_rul['max_cycle'] - test_df_with_rul['time_cycles']
    test_df_with_rul.drop(columns=['max_cycle'], inplace=True)
    
    # Step 2: Create advanced features (using the same sensor list as training)
    test_df_enhanced = test_df_with_rul.copy()
    
    # Group by each individual engine to apply time-series operations
    for unit_id, group in test_df_enhanced.groupby('unit_id'):
        for sensor in important_sensors:
            col_name = f'sensor_{sensor}'
            
            # Moving Average
            ma_col = f'{col_name}_MA_5'
            test_df_enhanced.loc[group.index, ma_col] = group[col_name].rolling(window=5, min_periods=1).mean()
            
            # Rolling Standard Deviation
            std_col = f'{col_name}_Rolling_Std'
            test_df_enhanced.loc[group.index, std_col] = group[col_name].rolling(window=5, min_periods=1).std()
            
            # Cumulative Sum
            cumsum_col = f'{col_name}_CumSum'
            test_df_enhanced.loc[group.index, cumsum_col] = group[col_name].cumsum()
    
    # Handle any NaN values
    test_df_enhanced.fillna(method='bfill', inplace=True)
    
    # Step 3: Normalize using the FITTED scaler from training
    all_numeric_columns = test_df_enhanced.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_exclude = ['unit_id', 'time_cycles', 'RUL']
    cols_to_normalize = [col for col in all_numeric_columns if col not in cols_to_exclude]
    
    test_df_enhanced[cols_to_normalize] = fitted_scaler.transform(test_df_enhanced[cols_to_normalize])
    
    print("✅ Test data preprocessing complete!")
    return test_df_enhanced

# =============================================================================
# UPDATED get_final_cycle_predictions function for LSTM
# =============================================================================

def get_final_cycle_predictions(test_df_processed, model, feature_columns, sequence_length=30):
    """
    For each engine, creates the proper sequence for the LSTM model
    and gets the prediction at its final cycle.
    """
    print("🔍 Creating sequences and extracting predictions for final cycle of each engine...")
    
    final_predictions = []
    engine_ids = []
    
    for unit_id in test_df_processed['unit_id'].unique():
        engine_data = test_df_processed[test_df_processed['unit_id'] == unit_id]
        engine_data = engine_data.sort_values('time_cycles')
        
        # We need at least sequence_length cycles to make a prediction
        if len(engine_data) < sequence_length:
            print(f"⚠️  Engine {unit_id} has only {len(engine_data)} cycles (need {sequence_length}), skipping...")
            continue
        
        # Get the last 'sequence_length' cycles for this engine
        final_sequence_data = engine_data[feature_columns].iloc[-sequence_length:]
        
        # Reshape to 3D: (1 sample, sequence_length timesteps, n_features)
        sequence_3d = final_sequence_data.values.reshape(1, sequence_length, len(feature_columns))
        
        # Make prediction
        prediction = model.predict(sequence_3d, verbose=0)[0][0]  # Get scalar value
        final_predictions.append(prediction)
        engine_ids.append(unit_id)
    
    return np.array(final_predictions), np.array(engine_ids)

def main():
    """Main function to run the final evaluation."""
    print("🎯 PHASE 6: Final Model Evaluation on Test Set")
    print("==========================================================")
    
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    test_data_path = os.path.join(project_root, 'data', 'test_FD001.txt')
    true_rul_path = os.path.join(project_root, 'data', 'RUL_FD001.txt')
    scaler_path = os.path.join(project_root, 'models', 'fitted_scaler.pkl')
    model_path = os.path.join(project_root, 'models', 'lstm_model.keras')
    
    # Load the true RUL values
    true_rul_df = load_true_rul_data(true_rul_path)
    true_rul = true_rul_df['True_RUL'].values
    
    # Load the fitted scaler from training
    print("📂 Loading fitted scaler...")
    fitted_scaler = joblib.load(scaler_path)
    
    # Load the champion model
    print("📂 Loading champion LSTM model...")
    champion_model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess test data
    raw_test_df = load_raw_test_data(test_data_path)
    important_sensors = [4, 7, 11, 12, 15]  # Same as training
    
    test_df_processed = preprocess_test_data(raw_test_df, fitted_scaler, important_sensors)
    
    # Get feature columns (same as training)
    feature_columns = [col for col in test_df_processed.columns 
                      if col not in ['unit_id', 'time_cycles', 'RUL']]
    
    # Get predictions for final cycle of each engine
    test_predictions, engine_ids = get_final_cycle_predictions(
        test_df_processed, champion_model, feature_columns
    )
    
    # Calculate final metrics
    final_rmse = np.sqrt(mean_squared_error(true_rul, test_predictions))
    final_mae = mean_absolute_error(true_rul, test_predictions)
    final_r2 = r2_score(true_rul, test_predictions)
    
    # Print comprehensive results
    print("\n" + "="*60)
    print("FINAL MODEL PERFORMANCE ON TEST SET")
    print("="*60)
    print(f"Test Set RMSE: {final_rmse:.2f} cycles")
    print(f"Test Set MAE: {final_mae:.2f} cycles")
    print(f"Test Set R²: {final_r2:.4f}")
    print("="*60)
    
    # Save results for reporting
    results_df = pd.DataFrame({
        'Engine_ID': engine_ids,
        'True_RUL': true_rul,
        'Predicted_RUL': test_predictions,
        'Absolute_Error': np.abs(true_rul - test_predictions)
    })
    
    results_path = os.path.join(project_root, 'results', 'final_test_results.csv')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"💾 Detailed results saved to: {results_path}")
    
    return final_rmse, final_mae, final_r2

if __name__ == "__main__":
    # Add TensorFlow import here to avoid conflicts
    import tensorflow as tf
    tf.random.set_seed(RANDOM_STATE)
    
    main()