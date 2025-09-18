# Data Dictionary

## Original Data Files
| File Name        | Description                                           | Records  | Columns |
|------------------|-------------------------------------------------------|----------|---------|
| train_FD001.txt  | Training data for 100 engines (until failure)          | ~20.6k   | 26      |
| test_FD001.txt   | Test data for 100 engines (truncated before failure)   | ~13.0k   | 26      |
| RUL_FD001.txt    | True Remaining Useful Life for test engines            | 100      | 1       |

---

## Column Descriptions
| Column Name      | Description                          | Units/Notes       |
|------------------|--------------------------------------|------------------|
| unit_id          | Unique identifier for each engine    | Integer          |
| time_cycles      | Number of operational cycles elapsed | Integer          |
| op_setting_1–3   | Operational settings                 | Normalized       |
| sensor_1–21      | Sensor readings (vibration, temp, etc.) | Normalized   |

---

## Engineered Features (Post-Processing)
| Feature Name            | Description                                | Rationale                          |
|--------------------------|--------------------------------------------|------------------------------------|
| RUL                     | Target variable: Remaining Useful Life     | `max_cycle - current_cycle`        |
| sensor_X_MA_5           | 5-cycle moving average of sensor X         | Smooths out noise                  |
| sensor_X_Rolling_Std    | Rolling standard deviation of sensor X     | Captures volatility (wear patterns)|
| sensor_X_CumSum         | Cumulative sum of sensor X                 | Tracks accumulated degradation     |
