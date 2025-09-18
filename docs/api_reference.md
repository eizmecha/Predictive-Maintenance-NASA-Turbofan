# API Reference

This document provides an overview of the main functions available in the project’s Python modules.

---

## Module: `src.data.preprocessing`
| Function                     | Description                                | Parameters                        | Returns                   |
|------------------------------|--------------------------------------------|-----------------------------------|---------------------------|
| `add_rul_column(df)`         | Calculates and adds the RUL column         | `df`: Raw DataFrame               | DataFrame with RUL        |
| `create_advanced_features(df, sensor_list)` | Creates moving averages, rolling std, etc. | `df`: DataFrame, `sensor_list`: list of sensors | Enhanced DataFrame |
| `prepare_train_data(df, sensor_list)` | Full preprocessing pipeline for training data | `df`: Raw DataFrame, `sensor_list`: list | `(processed_df, scaler)` |

---

## Module: `src.utils.metrics`
| Function             | Description                                  |
|----------------------|----------------------------------------------|
| `calculate_metrics(y_true, y_pred)` | Computes RMSE, MAE, R² metrics. Returns dict. |
| `print_metrics(metrics_dict, model_name)` | Nicely prints evaluation metrics. |

---

## Module: `src.models.train_model`
| Function             | Description                                  |
|----------------------|----------------------------------------------|
| `train_lstm_model()` | Trains an LSTM model for RUL prediction.     |
| `train_rf_model()`   | Trains a Random Forest model.                |
| `train_xgb_model()`  | Trains an XGBoost model.                     |
