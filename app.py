# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
import json

# Config
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("🔧 NASA Turbofan Predictive Maintenance Dashboard")
st.markdown("Interactive dashboard for predicting equipment failures")

# ===== Load data and models =====
try:
    df = pd.read_csv("synthetic_data.csv")
    feature_names = joblib.load("feature_names.joblib")
    target_names = ["Normal", "Failure"]

    try:
        rf_model = joblib.load("random_forest_model.joblib")
        xgb_model = joblib.load("xgboost_model.joblib")
        nn_model = tf.keras.models.load_model("neural_network_model.h5")
        scaler = joblib.load("scaler.joblib")

        with open("model_results.json", "r") as f:
            results = json.load(f)

        cm_df = pd.read_csv("confusion_matrices.csv")

        models_loaded = True
    except:
        st.warning("⚠️ Models not found. Please run main.py first.")
        models_loaded = False
        rf_model, xgb_model, nn_model, scaler, results, cm_df = None, None, None, None, {}, None

except FileNotFoundError:
    st.error("❌ synthetic_data.csv not found. Please run main.py first.")
    st.stop()

# ===== Sidebar inputs =====
st.sidebar.header("🔧 Machine Input Parameters")

user_input = {}
for i, feature in enumerate(feature_names[:10]):  # demo on first 10 features
    user_input[feature] = st.sidebar.slider(
        f"{feature}",
        float(df[feature].min()),
        float(df[feature].max()),
        float(df[feature].mean())
    )

input_df = pd.DataFrame([user_input])

model_choice = st.sidebar.selectbox(
    "Select Model:", ["Random Forest", "XGBoost", "Neural Network"]
)

# ===== Prediction =====
if st.sidebar.button("🚀 Predict Failure") and models_loaded:
    try:
        if model_choice == "Random Forest":
            prediction = rf_model.predict(input_df)[0]
            probability = rf_model.predict_proba(input_df)[0][1]
        elif model_choice == "XGBoost":
            prediction = xgb_model.predict(input_df)[0]
            probability = xgb_model.predict_proba(input_df)[0][1]
        else:  # Neural Network
            scaled_input = scaler.transform(input_df.values)
            probability = nn_model.predict(scaled_input, verbose=0)[0][0]
            prediction = 1 if probability > 0.5 else 0

        st.sidebar.success(f"Prediction: {target_names[prediction]}")
        st.sidebar.info(f"Failure Probability: {probability:.3f}")

    except Exception as e:
        st.sidebar.error(f"Prediction error: {e}")

# ===== Tabs =====
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Overview",
    "📈 Analytics",
    "🤖 Model Performance",
    "📋 Sample Predictions"
])

with tab1:
    st.header("Dataset Overview")
    st.dataframe(df.head(10))
    st.write(f"Dataset shape: {df.shape}")
    st.write(f"Number of normal operations: {(df['target']==0).sum()}")
    st.write(f"Number of failures: {(df['target']==1).sum()}")

with tab2:
    st.header("Data Analytics")
    col1, col2 = st.columns(2)

    with col1:
        # Target distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        target_counts = df["target"].value_counts()
        ax.pie(target_counts, labels=target_names, autopct="%1.1f%%",
               colors=["lightblue", "lightcoral"])
        ax.set_title("Target Class Distribution")
        st.pyplot(fig)

    with col2:
        # Correlation heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        corr_matrix = df[feature_names[:5] + ["target"]].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig)

with tab3:
    st.header("Model Performance Comparison")
    if models_loaded:
        performance_data = {
            "Model": list(results.keys()),
            "Accuracy": list(results.values())
        }
        perf_df = pd.DataFrame(performance_data)
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(perf_df.set_index("Model"))
        with col2:
            st.dataframe(perf_df)

        # Show confusion matrices
        st.subheader("Confusion Matrices")
        st.dataframe(cm_df)

    else:
        st.info("Run main.py to train models and see performance metrics")

with tab4:
    st.header("Sample Predictions")
    sample_data = df[feature_names].sample(5)
    st.write("Sample sensor readings:")
    st.dataframe(sample_data)

    if models_loaded and st.button("Predict on Sample Data"):
        predictions = []
        for i, (idx, row) in enumerate(sample_data.iterrows()):
            input_row = pd.DataFrame([row])
            try:
                if model_choice == "Random Forest":
                    pred = rf_model.predict(input_row)[0]
                    proba = rf_model.predict_proba(input_row)[0][1]
                elif model_choice == "XGBoost":
                    pred = xgb_model.predict(input_row)[0]
                    proba = xgb_model.predict_proba(input_row)[0][1]
                else:
                    scaled_input = scaler.transform(input_row.values.reshape(1, -1))
                    proba = nn_model.predict(scaled_input, verbose=0)[0][0]
                    pred = 1 if proba > 0.5 else 0

                predictions.append({
                    "Sample": i + 1,
                    "Prediction": target_names[pred],
                    "Probability": f"{proba:.3f}",
                    "Actual": target_names[df.loc[idx, "target"]]
                })
            except Exception as e:
                predictions.append({
                    "Sample": i + 1,
                    "Prediction": "Error",
                    "Probability": "N/A",
                    "Actual": "Unknown"
                })

        predictions_df = pd.DataFrame(predictions)
        st.dataframe(predictions_df)

# ===== Footer =====
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Scikit-learn, XGBoost, and TensorFlow")