import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler

# Config
st.set_page_config(page_title="NASA Turbofan RUL Predictor", layout="wide")
st.title("✈️ NASA Turbofan Engine Predictive Maintenance Dashboard")
st.markdown("Predict Remaining Useful Life (RUL) of aircraft engines using advanced machine learning")

# Add custom CSS for the exit button
st.markdown("""
<style>
    .exit-button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        margin-top: 2rem;
    }
    .exit-button:hover {
        background-color: #ff2b2b;
        color: white;
    }
    .exit-message {
        text-align: center;
        padding: 2rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Session state to track if user wants to exit
if 'exit_app' not in st.session_state:
    st.session_state.exit_app = False

# Exit function
def exit_application():
    """Function to exit the application"""
    st.session_state.exit_app = True

# Check if user wants to exit first
if st.session_state.exit_app:
    st.balloons()
    st.markdown('<div class="exit-message">', unsafe_allow_html=True)
    st.success("✅ Thank you for using the NASA Turbofan RUL Predictor!")
    st.markdown("### 🚪 Application Closed")
    st.info("You can safely close this browser tab now.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ===== Load data and models =====
@st.cache_resource
def load_models_and_data():
    """Load trained models and necessary data"""
    try:
        # Get project root
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        # Load processed training data for reference
        train_data_path = os.path.join(project_root, 'data', 'processed', 'train_FD001_processed.csv')
        train_df = pd.read_csv(train_data_path)
        
        # Load test data for demo
        test_data_path = os.path.join(project_root, 'data', 'test_FD001.txt')
        column_names = ['unit_id', 'time_cycles'] + [f'op_setting_{i}' for i in range(1,4)] + [f'sensor_{i}' for i in range(1,22)]
        test_df = pd.read_csv(test_data_path, sep=r'\s+', header=None, names=column_names)
        
        # Load models
        lstm_model = tf.keras.models.load_model(os.path.join(project_root, 'models', 'lstm_model.keras'))
        rf_model = joblib.load(os.path.join(project_root, 'models', 'optimized_random_forest_model.pkl'))
        scaler = joblib.load(os.path.join(project_root, 'models', 'fitted_scaler.pkl'))
        
        # Load final test results
        results_path = os.path.join(project_root, 'results', 'final_test_results.csv')
        if os.path.exists(results_path):
            test_results = pd.read_csv(results_path)
        else:
            test_results = None
            
        return {
            'train_df': train_df,
            'test_df': test_df,
            'lstm_model': lstm_model,
            'rf_model': rf_model,
            'scaler': scaler,
            'test_results': test_results,
            'success': True
        }
        
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return {'success': False}

# Load everything
resources = load_models_and_data()

if not resources['success']:
    st.error("❌ Failed to load required files. Please run the training pipeline first.")
    st.stop()

# Extract resources
train_df = resources['train_df']
test_df = resources['test_df']
lstm_model = resources['lstm_model']
rf_model = resources['rf_model']
scaler = resources['scaler']
test_results = resources['test_results']

# ===== Sidebar inputs =====
st.sidebar.header("🔧 Engine Selection & Parameters")

# Engine selection - Calculate actual total cycles for each engine
engine_cycles = test_df.groupby('unit_id')['time_cycles'].max().reset_index()
engine_cycles.columns = ['unit_id', 'total_cycles']
engine_ids_with_cycles = [(f"Engine {id} ({cycles} cycles)", id) for id, cycles in zip(engine_cycles['unit_id'], engine_cycles['total_cycles'])]

selected_engine_option = st.sidebar.selectbox(
    "Select Engine", 
    options=[opt[0] for opt in engine_ids_with_cycles],
    index=0
)

# Extract the actual engine ID from the selection
selected_engine = int(selected_engine_option.split(' ')[1])  # Gets the number from "Engine X (Y cycles)"

# Get selected engine data
engine_data = test_df[test_df['unit_id'] == selected_engine].copy()

# Calculate actual values
total_cycles = engine_data['time_cycles'].max()
current_cycle = engine_data['time_cycles'].iloc[-1]  # Last recorded cycle

# Model selection
model_choice = st.sidebar.radio(
    "Select Prediction Model:",
    ["LSTM Neural Network", "Optimized Random Forest"],
    help="LSTM: Advanced deep learning for sequence data | Random Forest: Robust traditional ML"
)

# Display engine info - FIXED: Show actual different values
st.sidebar.subheader("Engine Information")
st.sidebar.write(f"**Engine ID:** {selected_engine}")
st.sidebar.write(f"**Total Cycles:** {total_cycles}")
st.sidebar.write(f"**Current Cycle:** {current_cycle}")
st.sidebar.write(f"**Data Points:** {len(engine_data)}")

# ===== Preprocessing functions =====
def preprocess_engine_data(raw_engine_data, scaler, important_sensors):
    """Preprocess engine data for prediction"""
    # Create a copy to avoid modifying original
    engine_data = raw_engine_data.copy()
    
    # Calculate max cycle for this engine
    max_cycle = engine_data['time_cycles'].max()
    engine_data['RUL'] = max_cycle - engine_data['time_cycles']
    
    # Create advanced features (same as training)
    for sensor in important_sensors:
        col_name = f'sensor_{sensor}'
        
        # Moving Average
        engine_data[f'{col_name}_MA_5'] = engine_data[col_name].rolling(window=5, min_periods=1).mean()
        
        # Rolling Standard Deviation
        engine_data[f'{col_name}_Rolling_Std'] = engine_data[col_name].rolling(window=5, min_periods=1).std()
        
        # Cumulative Sum
        engine_data[f'{col_name}_CumSum'] = engine_data[col_name].cumsum()
    
    # Handle NaN values
    engine_data.fillna(method='bfill', inplace=True)
    
    # Identify columns to normalize
    all_numeric_columns = engine_data.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_exclude = ['unit_id', 'time_cycles', 'RUL']
    cols_to_normalize = [col for col in all_numeric_columns if col not in cols_to_exclude]
    
    # Normalize using the fitted scaler
    engine_data[cols_to_normalize] = scaler.transform(engine_data[cols_to_normalize])
    
    return engine_data, cols_to_normalize

# ===== Prediction =====
if st.sidebar.button("🚀 Predict RUL", type="primary"):
    with st.spinner("Processing engine data and making prediction..."):
        try:
            important_sensors = [4, 7, 11, 12, 15]
            processed_data, feature_columns = preprocess_engine_data(engine_data, scaler, important_sensors)
            
            if model_choice == "LSTM Neural Network":
                # Use last 30 cycles for LSTM prediction
                if len(processed_data) >= 30:
                    sequence_data = processed_data[feature_columns].iloc[-30:]
                    sequence_3d = sequence_data.values.reshape(1, 30, len(feature_columns))
                    prediction = lstm_model.predict(sequence_3d, verbose=0)[0][0]
                    
                    # Display prediction
                    st.sidebar.success(f"**Predicted RUL:** {prediction:.0f} cycles")
                    
                    # Status interpretation
                    if prediction > 50:
                        st.sidebar.info("✅ Engine in good condition")
                    elif prediction > 20:
                        st.sidebar.warning("⚠️ Monitor engine closely")
                    else:
                        st.sidebar.error("🚨 Maintenance required soon!")
                else:
                    st.sidebar.warning("Not enough data for LSTM prediction (need ≥30 cycles)")
                    
            else:  # Random Forest
                # Use final cycle for RF prediction
                final_cycle_data = processed_data[feature_columns].iloc[-1:]
                prediction = rf_model.predict(final_cycle_data)[0]
                
                st.sidebar.success(f"**Predicted RUL:** {prediction:.0f} cycles")
                
                # Status interpretation
                if prediction > 50:
                    st.sidebar.info("✅ Engine in good condition")
                elif prediction > 20:
                    st.sidebar.warning("⚠️ Monitor engine closely")
                else:
                    st.sidebar.error("🚨 Maintenance required soon!")
                    
        except Exception as e:
            st.sidebar.error(f"Prediction error: {str(e)}")

# ===== Tabs =====
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Engine Data",
    "📈 Sensor Analytics", 
    "🤖 Model Performance",
    "🔍 Prediction Analysis"
])

with tab1:
    st.header("Engine Sensor Data")
    st.write(f"Data for Engine {selected_engine} ({len(engine_data)} data points, {total_cycles} total cycles)")
    
    # Show raw sensor data - FIXED: use width instead of use_container_width
    st.dataframe(engine_data.head(10), width='stretch')
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cycles", total_cycles)
    with col2:
        st.metric("Current Cycle", current_cycle)
    with col3:
        st.metric("Engine ID", selected_engine)
    with col4:
        st.metric("Data Points", len(engine_data))

with tab2:
    st.header("Sensor Trends and Analytics")
    
    # Sensor selection
    sensor_options = [f'Sensor {i}' for i in [4, 7, 11, 12, 15]]
    selected_sensors = st.multiselect("Select sensors to visualize:", sensor_options, default=sensor_options[:3])
    
    if selected_sensors:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for sensor in selected_sensors:
            sensor_num = int(sensor.split(' ')[1])
            ax.plot(engine_data['time_cycles'], engine_data[f'sensor_{sensor_num}'], 
                   label=f'Sensor {sensor_num}', linewidth=2)
        
        ax.set_xlabel('Time Cycles')
        ax.set_ylabel('Sensor Value')
        ax.set_title(f'Engine {selected_engine} - Sensor Trends')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Sensor Correlations")
    sensor_cols = [f'sensor_{i}' for i in [4, 7, 11, 12, 15]]
    if len(engine_data) > 1:
        corr_matrix = engine_data[sensor_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax, fmt='.2f')
        ax.set_title("Sensor Correlation Heatmap")
        st.pyplot(fig)

with tab3:
    st.header("Model Performance Metrics")
    
    if test_results is not None:
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test RMSE", "28.33 cycles")
        with col2:
            st.metric("Test MAE", "18.22 cycles") 
        with col3:
            st.metric("R² Score", "0.535")
        
        # True vs Predicted plot
        st.subheader("Model Performance on Test Set")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(test_results['True_RUL'], test_results['Predicted_RUL'], alpha=0.6)
        ax.plot([0, 150], [0, 150], 'r--', label='Perfect Prediction')
        ax.set_xlabel('True RUL (cycles)')
        ax.set_ylabel('Predicted RUL (cycles)')
        ax.set_title('True vs Predicted RUL (Test Set)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Error distribution
        st.subheader("Prediction Error Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        errors = test_results['True_RUL'] - test_results['Predicted_RUL']
        ax.hist(errors, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(errors.mean(), color='red', linestyle='--', label=f'Mean Error: {errors.mean():.1f} cycles')
        ax.set_xlabel('Prediction Error (cycles)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Prediction Errors')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Final test results not available. Run final_evaluation.py to generate performance metrics.")

with tab4:
    st.header("Advanced Prediction Analysis")
    
    # Feature importance (for Random Forest)
    if hasattr(rf_model, 'feature_importances_'):
        st.subheader("Feature Importance (Random Forest)")
        feature_importance = pd.DataFrame({
            'feature': train_df.columns.drop(['unit_id', 'time_cycles', 'RUL']),
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(feature_importance['feature'], feature_importance['importance'])
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Most Important Features')
        st.pyplot(fig)
    
    # Model comparison - FIXED: use width instead of use_container_width
    st.subheader("Model Comparison")
    comparison_data = pd.DataFrame({
        'Model': ['LSTM Neural Network', 'Optimized Random Forest'],
        'RMSE': [28.33, 31.72],
        'R² Score': [0.535, 0.480]
    })
    
    st.dataframe(comparison_data, width='stretch')

# ===== Footer =====
st.markdown("---")
st.markdown("### 🎯 Project Summary")
st.markdown("""
This predictive maintenance system uses advanced machine learning to forecast the Remaining Useful Life (RUL) 
of NASA turbofan engines. The models were trained on real sensor data and can predict failures with 
**~28 cycle accuracy**, providing ample warning for maintenance planning.

**Key Features:**
- ✈️ **Real-time RUL predictions** for individual engines
- 📊 **Interactive sensor data visualization**
- 🤖 **Multiple model comparison** (LSTM vs Random Forest)
- 📈 **Comprehensive performance analytics**
- 🔧 **Actionable maintenance recommendations**
""")

# Add exit button in the sidebar
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Exit Application", type="secondary", 
                    help="Close the NASA Turbofan RUL Predictor", 
                    on_click=exit_application):
    
    pass

st.markdown("---")
st.markdown("Built using Streamlit, TensorFlow, Scikit-learn, and NASA Turbofan Data")
