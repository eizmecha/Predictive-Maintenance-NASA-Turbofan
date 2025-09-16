"""
Baseline model training utilities
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_baseline_models(X_train, y_train, X_val, y_val):
    """Train multiple baseline models and return performance"""
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    performance = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        performance[name] = {
            'Model': model,
            'RMSE': mean_squared_error(y_val, y_pred, squared=False),
            'MAE': mean_absolute_error(y_val, y_pred),
            'R2': r2_score(y_val, y_pred)
        }
    
    return performance