# Predictive Maintenance for Turbofan Engines - Final Project Report

## Abstract
This project develops a Predictive Maintenance system for NASA’s Turbofan Engine dataset (FD001 subset). Using machine learning and deep learning approaches (Linear Regression, Random Forest, XGBoost, and LSTM), the system predicts Remaining Useful Life (RUL) of engines. Results show that the LSTM model significantly outperforms traditional models, achieving a validation RMSE of ~12.6 cycles, though performance on the test set (RMSE ~28.9) reveals challenges such as overfitting and domain shift. The project demonstrates the potential of temporal deep learning models for real-world maintenance applications.

---

## 1. Introduction
### Problem Statement
Predictive maintenance aims to forecast equipment failures before they occur. In this project, we predict the **Remaining Useful Life (RUL)** of turbofan engines, i.e., the number of operational cycles left before failure.

### Dataset Overview
- **Source**: NASA C-MAPSS dataset  
- **Subset used**: FD001 (100 engines, 1 operating condition, 1 failure mode)  
- **Files**:  
  - `train_FD001.txt` → training data until failure  
  - `test_FD001.txt` → test data truncated before failure  
  - `RUL_FD001.txt` → true RUL labels for the test set  

### Project Objectives
- Preprocess sensor and operational data to extract meaningful features.  
- Train multiple models to predict RUL.  
- Evaluate models using RMSE, MAE, and R².  
- Deploy the best-performing model for inference on unseen test engines.  

---

## 2. Methodology
### Data Preprocessing
*Credit: Osamah*  
- Added RUL column: `RUL = max_cycle - current_cycle`  
- Normalized sensor values  
- Engineered features: moving averages, rolling std, cumulative sums  

### Exploratory Data Analysis (EDA)
*Credit: Ali*  
- Identified key sensors contributing to degradation.  
- Visualized sensor trends vs. engine cycles.  
- Observed patterns of monotonic decrease/increase for critical sensors.  

### Modeling Approach
- **Linear Regression** → baseline  
- **Random Forest, XGBoost** → non-linear baselines  
- **LSTM** → captures temporal sequences of sensor data  

### Validation Strategy
- Training/validation split from training set.  
- RMSE as primary metric.  
- Final evaluation on test set with provided RUL labels.  

---

## 3. Results & Discussion
### Baseline Model Performance (Validation)
| Model              | RMSE  | MAE  | R²   |
|--------------------|-------|------|------|
| Linear Regression  | 38.98 | 30.1 | 0.667 |
| Random Forest      | 31.50 | 21.9 | 0.783 |
| XGBoost            | 34.28 | 24.2 | 0.743 |
| **LSTM**           | **12.59** | **8.65** | **0.958** |

✅ LSTM outperformed all others due to sequence learning ability.

### Final Test Performance
- **MAE**: 21.52  
- **RMSE**: 28.90  
- **R²**: 0.516  

### Discussion
- Validation results were excellent; test performance dropped.  
- Possible reasons: overfitting, data distribution shift, noise in test set.  
- LSTM remains the champion model despite generalization challenges.  

---

## 4. Conclusion & Future Work
### Conclusion
This project successfully demonstrated predictive maintenance for turbofan engines using ML/DL. The LSTM achieved state-of-the-art performance on validation, proving the strength of temporal models.

### Future Improvements
- Add more regularization to reduce overfitting.  
- Explore hybrid CNN-LSTM or Transformer models.  
- Test on additional subsets (FD002–FD004).  
- Deploy the trained model as a real-time API.  
