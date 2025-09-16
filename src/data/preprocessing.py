import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import joblib

def add_rul_column(df):
    # Calculate the maximum cycle each engine reaches before failure
    max_cycle_per_engine = df.groupby('unit_id')['time_cycles'].max().reset_index()
    max_cycle_per_engine.rename(columns={'time_cycles': 'max_cycle'}, inplace=True)

    # Merge this information back to the original dataframe
    df_with_rul = df.merge(max_cycle_per_engine, on='unit_id', how='left')

    # Perform the RUL calculation
    df_with_rul['RUL'] = df_with_rul['max_cycle'] - df_with_rul['time_cycles']

    # Clean up by removing the temporary 'max_cycle' column
    df_with_rul.drop(columns=['max_cycle'], inplace=True)

    return df_with_rul

def create_advanced_features(df, sensor_list):
    df_enhanced = df.copy()

    # Group by each individual engine to apply time-series operations correctly
    for unit_id, group in df_enhanced.groupby('unit_id'):
        # For each sensor deemed critical
        for sensor in sensor_list:
            col_name = f'sensor_{sensor}'

            # 1. Smooth the signal with a Moving Average (window=5)
            ma_col = f'{col_name}_MA_5'
            df_enhanced.loc[group.index, ma_col] = group[col_name].rolling(window=5, min_periods=1).mean()

            # 2. Capture operational volatility with Rolling Standard Deviation
            std_col = f'{col_name}_Rolling_Std'
            df_enhanced.loc[group.index, std_col] = group[col_name].rolling(window=5, min_periods=1).std()

            # 3. Quantify total degradation exposure with Cumulative Sum
            cumsum_col = f'{col_name}_CumSum'
            df_enhanced.loc[group.index, cumsum_col] = group[col_name].cumsum()

    # Any NaN values created at the start of each engine's rolling window are filled via backpropagation
    df_enhanced.fillna(method='bfill', inplace=True)

    return df_enhanced

def normalize_data(df, columns_to_normalize):
    scaler = MinMaxScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df, scaler

def prepare_train_data(df, important_sensors):
    print("Initiating Data Preprocessing Pipeline...")
    print("Step 1: Engineering target variable 'RUL'...")
    df_processed = add_rul_column(df)

    print("Step 2: Creating advanced features (Moving Averages, Volatility, Cumulative Sums)...")
    df_processed = create_advanced_features(df_processed, important_sensors)

    # Identify all numeric columns for potential normalization
    all_numeric_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude identifiers, time, and the target variable from normalization
    cols_to_exclude = ['unit_id', 'time_cycles', 'RUL']
    cols_to_normalize = [col for col in all_numeric_columns if col not in cols_to_exclude]

    print("Step 3: Normalizing features to a [0, 1] range...")
    df_processed, fitted_scaler = normalize_data(df_processed, cols_to_normalize)

    print("Preprocessing complete! The data is now ready for modeling.")
    
    # FIXED: Use the correct variable name 'fitted_scaler' instead of 'scaler'
    # Save the fitted scaler for later use on test data
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scaler_path = os.path.join('models', 'fitted_scaler.pkl')
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(fitted_scaler, scaler_path)  # FIXED: changed 'scaler' to 'fitted_scaler'
    print(f"💾 Fitted scaler saved to: {scaler_path}")
    
    return df_processed, fitted_scaler

def load_raw_data(file_path):
    column_names = ['unit_id', 'time_cycles'] + [f'op_setting_{i}' for i in range(1,4)] + [f'sensor_{i}' for i in range(1,22)]
    
    try:
        # FIXED: Use raw string to avoid escape sequence warning
        df = pd.read_csv(file_path, sep=r'\s+', header=None, names=column_names)
        print(f"Raw data loaded successfully from: {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        print("Please make sure the raw data file exists.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df, fit_scaler=True, scaler=None):
    """
    Main preprocessing function that can be used for both training and inference.
    For training: fit_scaler=True, scaler=None
    For inference: fit_scaler=False, scaler=previously_fitted_scaler
    """
    important_sensors = [4, 7, 11, 12, 15]
    
    # Add RUL column
    df_processed = add_rul_column(df)
    
    # Create advanced features
    df_processed = create_advanced_features(df_processed, important_sensors)
    
    # Identify columns to normalize
    all_numeric_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_exclude = ['unit_id', 'time_cycles', 'RUL']
    cols_to_normalize = [col for col in all_numeric_columns if col not in cols_to_exclude]
    
    # Normalize data
    if fit_scaler:
        df_processed, fitted_scaler = normalize_data(df_processed, cols_to_normalize)
        return df_processed, fitted_scaler
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when fit_scaler=False")
        df_processed[cols_to_normalize] = scaler.transform(df_processed[cols_to_normalize])
        return df_processed, scaler

if __name__ == "__main__":
    """Main function to run the complete preprocessing pipeline"""
    print("Starting Data Preprocessing Pipeline")
    print("=" * 50)
    
    # Get the absolute path to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define absolute paths
    raw_data_path = os.path.join('data', 'train_FD001.txt')
    processed_data_path = os.path.join('data','processed', 'train_FD001_processed.csv')
    
    print(f"Looking for data at: {raw_data_path}")
    
    # Load raw data
    train_df = load_raw_data(raw_data_path)
    if train_df is None:
        exit(1)
    
    print(f"Raw data shape: {train_df.shape}")
    
    # Process data
    important_sensors = [4, 7, 11, 12, 15]
    processed_df, scaler = prepare_train_data(train_df, important_sensors)
    
    # Create processed directory if it doesn't exist
    processed_dir = os.path.dirname(processed_data_path)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save processed data
    processed_df.to_csv(processed_data_path, index=False)
    
    print(f"Processed data saved to: {processed_data_path}")
    print(f"Processed data shape: {processed_df.shape}")
    print("Preprocessing complete! Ready for model training.")