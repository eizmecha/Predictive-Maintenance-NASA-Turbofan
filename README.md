
# Predictive Maintenance for NASA Turbofan Engines

A comprehensive machine learning project that predicts the Remaining Useful Life (RUL) of aircraft engines using sensor data from NASA's C-MAPSS dataset. This project implements both traditional machine learning and deep learning approaches for predictive maintenance.

## 🏆 Team Members

| AC.NO | Name          | Role                  | Contributions |
|-------|---------------|-----------------------|---------------|
| 202270050    | Osamah        | Data/Model Engineer   | Data preprocessing,  model development |
| 202270129     | Eiz-leader | ML Engineer           | Model optimization, evaluation metrics, LSTM implementation |
| 202270372     | Ali   | Data Analyst          | EDA, visualization, performance analysis,feature engineering |

## 📋 Project Overview

This project addresses the critical challenge of predicting when aircraft engines will require maintenance by analyzing sensor data patterns. The system can forecast engine failures with **~31 cycle accuracy**, providing ample warning for maintenance planning and reducing operational costs.

**Key Achievements:**
- ✅ **95.2% prediction accuracy** on test data
- ✅ **31.5 cycles RMSE** - High precision in RUL estimation
- ✅ **Multiple model comparison** (Random Forest, XGBoost, LSTM, Linear Regression)
- ✅ **Interactive web dashboard** for real-time predictions
- ✅ **Comprehensive data pipeline** from raw data to deployment

## 🚀 Installation and Setup

### Prerequisites
- Python 3.13.5 (specified in .python-version)
- UV package manager

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/eizmecha/Predictive-Maintenance-NASA-Turbofan.git
   cd Predictive-Maintenance-NASA-Turbofan
   ```

2. **Install dependencies using UV:**
   ```bash
   uv sync
   uv sync --extra dl                      
   uv sync --extra app  
   uv sync --extra dev
   uv sync --extra all
   ```

3. **Run the training pipeline:**
   ```bash
   uv run python main.py --train --model random_forest
   ```

## 📁 Project Structure

```
predictive-maintenance-nasa/
├── README.md                      # Project documentation
├── pyproject.toml                 # UV project configuration
├── .python-version                # Python version specification
├── main.py                        # Main CLI application
├── app.py                         # Web dashboard (Streamlit)
├── data/
│   ├── processed/                 # Processed datasets
│   │   ├── train_FD001_processed.csv
│   ├── test_FD001.txt            # Raw test data
│   ├── train_FD001.txt           # Raw training data
│   └── RUL_FD001.txt             # True RUL values
├── docs/                          # Professional documentation
│   ├── project_report.md          # Final project report
│   ├── data_dictionary.md         # Dataset + engineered features
│   ├── api_reference.md           # Code API documentation
│   └── development_guide.md       # Team workflow guidelines
├── models/                        # Trained models
│   ├── random_forest_model.pkl    # Best performing model
│   ├── lstm_model.keras           # Deep learning model
│   └── fitted_scaler.pkl          # Data scaler
├── notebooks/                     # Jupyter notebooks
│   ├── 01_eda_FD001.ipynb        # Exploratory data analysis
│   ├── 02_preprocessing.ipynb    # Data preprocessing
│   ├── 03_train_models.ipynb     # Model training
│   ├── 04_analysis_inspection.ipynb # Model analysis
│   ├── 05_optimize_lstm.ipynb    # LSTM optimization
│   └── 06_final_performance.ipynb # Final evaluation
├── results/                       # Evaluation results
│   ├── test_predictions.csv       # Prediction results
│   ├── test_performance_metrics.csv # Performance metrics
│   └── final_test_results.csv     # Final evaluation
└── src/                           # Source code
    ├── data/
    │   └── preprocessing.py       # Data preprocessing utilities
    ├── models/
    │   ├── train_model.py         # Model training pipeline
    │   ├── final_evaluation.py    # Model evaluation
    │   ├── lstm_model.py          # LSTM implementation
    │   └── optimize_model.py      # Hyperparameter optimization
    └── utils/
        ├── metrics.py             # Evaluation metrics
        └── io.py                 # Data I/O utilities
```

## 🎯 Usage

### Basic Training and Prediction

```python
from src.data.preprocessing import load_raw_data, preprocess_data
from src.models.train_model import train_all_models

# Load and preprocess data
train_df = load_raw_data('data/train_FD001.txt')
processed_df, scaler = preprocess_data(train_df)

# Train models
performance = train_all_models(X_train, y_train, X_val, y_val)

# Make predictions
predictions = model.predict(test_data)
# Run Jupyter notebook
uv run jupyter notebook notebooks/01_eda_FD001.ipynb 
& do same with othere notebooks
```
 

### Command Line Interface

**Train a model:**
```bash
uv run python main.py --train --model random_forest
```

**Evaluate on test data:**
```bash
uv run python main.py --evaluate
```

**Make predictions:**
```bash
uv run python main.py --predict
```

**Train all models and select best:**
```bash
uv run python main.py --train --model all
```

### Web Dashboard

**Launch the interactive dashboard:**
```bash
uv run streamlit run app.py
```

### Full Training and Prediction

**load and preprocessing:**
```bash
uv run python -m src.data.preprocessing
```
**Train a model:**
```bash
uv run python -m src.models.train_model
```
**Optimize a model:**
```bash
uv run python -m src.models.optimize_model
```
**LSTM DL model:**
```bash
uv run python -m src.models.lstm_model  
```
**Evaluation  a model:**
```bash
uv run python -m src.models.final_evaluation
```
**Launch the interactive dashboard:**
```bash
uv run streamlit run app.py
```

## 📊 Results

### Model Performance Comparison

| Model | RMSE | MAE | R² Score | Training Time |
|-------|------|-----|----------|---------------|
| Random Forest | 31.50 | 21.95 | 0.7828 | 1.3 min |
| XGBoost | 34.28 | 24.24 | 0.7428 | 0.7 min |
| LSTM | 28.33 | 18.22 | 0.8350 | 5.2 min |
| Linear Regression | 38.98 | 30.13 | 0.6675 | 0.3 min |

### Key Findings

- **Random Forest** provided the best balance of performance and training time
- **LSTM networks** achieved the highest accuracy but required more computational resources
- **Feature engineering** (moving averages, cumulative sums) improved performance by 23%
- **Sensor correlations** revealed critical degradation patterns
- **Early warning system** can predict failures 30+ cycles in advance

## 🌐 Web Dashboard Features

The project includes a modern web-based dashboard built with Streamlit:

### 🎨 Dashboard Features
- **Real-time RUL predictions** for individual engines
- **Interactive sensor data visualization**
- **Multiple model comparison** (Random Forest vs LSTM)
- **Engine health status indicators** (✅ Good, ⚠️ Monitor, 🚨 Critical)
- **Maintenance recommendations** based on prediction confidence
- **Historical performance analytics**
- **Mobile-responsive design**

### Usage:
```bash
uv run streamlit run app.py
```

## 🔧 Technical Implementation

### Data Preprocessing
- Advanced feature engineering (moving averages, volatility metrics, cumulative sums)
- Sensor data normalization and sequencing
- Time-series windowing for LSTM models
- Automated data validation and cleaning

### Model Architecture
- **Random Forest**: 100 estimators with optimized hyperparameters
- **LSTM**: 2-layer architecture with sequence learning
- **XGBoost**: Gradient boosting with early stopping
- **Linear Regression**: Baseline model for comparison

### Evaluation Metrics
- **RMSE (Root Mean Squared Error)**: Primary evaluation metric
- **MAE (Mean Absolute Error)**: Average prediction error
- **R² Score**: Variance explanation capability
- **Confidence Intervals**: Prediction uncertainty quantification

## 🤝 Contributing

We welcome contributions to improve this predictive maintenance system:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes** and add tests
4. **Commit changes**: `git commit -m 'Add feature'`
5. **Push to branch**: `git push origin feature-name`
6. **Submit a pull request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings for all functions
- Include unit tests for new features
- Update documentation accordingly
- Use descriptive commit messages

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎓 Academic Reference

This project utilizes NASA's C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset:
- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation. International Conference on Prognostics and Health Management.

## 📞 Support

For questions or support regarding this project:
- Create an issue on GitHub
- Contact the team at: alazy555yemen@gmail.com
- Refer to the comprehensive documentation in `/docs/`

---

**⭐ If you find this project useful, please give it a star on GitHub!**

