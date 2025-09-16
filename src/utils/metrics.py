import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def print_metrics(metrics_dict, model_name="Model"):
    """Formats and prints the metrics from calculate_metrics()."""
    print(f"--- {model_name} Performance ---")
    print(f"RMSE: {metrics_dict['RMSE']:.2f}")
    print(f"MAE: {metrics_dict['MAE']:.2f}")
    print(f"R2 Score: {metrics_dict['R2']:.4f}")