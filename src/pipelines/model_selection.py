import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.features.feature_engineering import create_features
from src.features.preprocess import build_preprocessor
from src.data.load_data import load_data


def find_best_threshold(y_true, probs):
    """Find best threshold based on F1 score"""
    best_f1 = 0
    best_threshold = 0.5

    for t in [0.5, 0.4, 0.3, 0.2]:
        preds = (probs > t).astype(int)
        score = f1_score(y_true, preds)

        if score > best_f1:
            best_f1 = score
            best_threshold = t

    return best_threshold, best_f1


def run_model_selection(data_path):

    # =========================
    # 1. LOAD DATA
    # =========================
    df = load_data(data_path)
    print("✅ Data loaded successfully")

    # =========================
    # 2. FEATURE ENGINEERING
    # =========================
    df = create_features(df)

    # =========================
    # 3. CLEAN DATA
    # =========================
    if "Payment Delay" in df.columns:
        df = df.drop("Payment Delay", axis=1)

    if "CustomerID" in df.columns:
        df = df.drop("CustomerID", axis=1)

    df = df.drop_duplicates()

    print("\nMissing values:\n", df.isnull().sum())

    # =========================
    # 4. SPLIT
    # =========================
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    preprocessor = build_preprocessor(X)

    # =========================
    # 5. MODELS
    # =========================
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    # =========================
    # 6. SPLIT DATA
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    results = {}

    # =========================
    # 7. TRAIN + EVALUATE
    # =========================
    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        # Predictions
        train_preds = pipeline.predict(X_train)
        test_probs = pipeline.predict_proba(X_test)[:, 1]

        # 🔥 FIND BEST THRESHOLD
        best_threshold, best_f1 = find_best_threshold(y_test, test_probs)

        # Apply best threshold
        test_preds = (test_probs > best_threshold).astype(int)

        # Metrics
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        roc = roc_auc_score(y_test, test_probs)

        results[name] = {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "f1": best_f1,
            "roc_auc": roc,
            "threshold": best_threshold
        }

        print("\n====================")
        print(name)
        print("====================")
        print(f"Train Accuracy : {train_acc:.4f}")
        print(f"Test Accuracy  : {test_acc:.4f}")
        print(f"Best F1 Score  : {best_f1:.4f}")
        print(f"ROC-AUC        : {roc:.4f}")
        print(f"Best Threshold : {best_threshold}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, test_preds))

        print("\nClassification Report:")
        print(classification_report(y_test, test_preds))

    # =========================
    # 8. SELECT BEST MODEL
    # =========================
    best_model_name = max(results, key=lambda x: results[x]["f1"])

    print("\n🏆 MODEL COMPARISON")
    for k, v in results.items():
        print(
            f"{k}: F1={v['f1']:.4f}, ROC={v['roc_auc']:.4f}, Threshold={v['threshold']}"
        )

    print(f"\n✅ Best Model: {best_model_name}")

    # =========================
    # 9. FINAL TRAIN + SAVE
    # =========================
    final_model = models[best_model_name]

    final_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", final_model)
    ])

    final_pipeline.fit(X_train, y_train)

    model_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../models")
    )
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "best_model.pkl")
    joblib.dump({
        "pipeline": final_pipeline,
        "threshold": results[best_model_name]["threshold"]
    }, model_path)

    print("\n💾 Best model + threshold saved successfully")

    return results