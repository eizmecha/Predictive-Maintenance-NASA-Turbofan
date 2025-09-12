# main.py
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import joblib
import json
import os


def create_synthetic_data(n_samples=5000, n_features=20, random_state=42):
    """Create synthetic predictive maintenance dataset"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        weights=[0.9, 0.1],  # class imbalance
        flip_y=0.01,
        random_state=random_state
    )

    feature_names = [f"sensor_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    df.to_csv("synthetic_data.csv", index=False)
    print("📊 Synthetic data created: synthetic_data.csv")

    return df, feature_names


def train_models(df, feature_names):
    """Train Random Forest, XGBoost, and Neural Network models + save results"""
    X = df[feature_names].values
    y = df["target"].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaler for Neural Network
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    confusion_data = []

    # ============ Random Forest ============
    rf_model = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_cm = confusion_matrix(y_test, rf_pred)
    joblib.dump(rf_model, "random_forest_model.joblib")
    results["Random Forest"] = rf_acc
    confusion_data.append(["Random Forest"] + rf_cm.flatten().tolist())
    print(f"✅ Random Forest saved. Accuracy: {rf_acc:.3f}")

    # ============ XGBoost ============
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_cm = confusion_matrix(y_test, xgb_pred)
    joblib.dump(xgb_model, "xgboost_model.joblib")
    results["XGBoost"] = xgb_acc
    confusion_data.append(["XGBoost"] + xgb_cm.flatten().tolist())
    print(f"✅ XGBoost saved. Accuracy: {xgb_acc:.3f}")

    # ============ Neural Network ============
    nn_model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(X_train_s.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    nn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    nn_model.fit(
        X_train_s, y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=64,
        verbose=1,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )
    nn_pred = (nn_model.predict(X_test_s, verbose=0) > 0.5).astype(int).flatten()
    nn_acc = accuracy_score(y_test, nn_pred)
    nn_cm = confusion_matrix(y_test, nn_pred)
    nn_model.save("neural_network_model.h5")
    results["Neural Network"] = nn_acc
    confusion_data.append(["Neural Network"] + nn_cm.flatten().tolist())
    print(f"✅ Neural Network saved. Accuracy: {nn_acc:.3f}")

    # Save scaler and feature names
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(feature_names, "feature_names.joblib")
    print("✅ Scaler and feature names saved.")
    # Save performance results
    with open("model_results.json", "w") as f:
        json.dump(results, f, indent=4)

    pd.DataFrame(confusion_data, columns=[
        "Model", "TN", "FP", "FN", "TP"
    ]).to_csv("confusion_matrices.csv", index=False)

    print("📑 Results saved: model_results.json & confusion_matrices.csv")


def main():
    print("=" * 60)
    print("🔧 NASA TURBOFAN PREDICTIVE MAINTENANCE PIPELINE")
    print("=" * 60)

    df, feature_names = create_synthetic_data()
    train_models(df, feature_names)

    print("🎉 Training complete. All models, scalers, and reports are saved.")


if __name__ == "__main__":
    main()