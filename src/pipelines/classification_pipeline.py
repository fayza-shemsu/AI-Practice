import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from src.features.preprocess import build_preprocessor
from src.data.load_data import load_data

def run_classification_pipeline(data_path):

    # =========================
    # 1. LOAD DATA
    # =========================
    df = load_data(data_path)
    print("✅ Data loaded successfully")

    # =========================
    # 2. REMOVE LEAKAGE / ID FEATURES
    # =========================
    if "Payment Delay" in df.columns:
        df = df.drop("Payment Delay", axis=1)

    if "CustomerID" in df.columns:
        df = df.drop("CustomerID", axis=1)

    # =========================
    # 3. QUICK CHECKS
    # =========================
    print(df["Churn"].unique())
    print(df.isnull().sum())
    print(df.duplicated().sum())

    # =========================
    # 4. SPLIT
    # =========================
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # =========================
    # 5. PREPROCESSOR
    # =========================
    preprocessor = build_preprocessor(X)

    # =========================
    # 6. MODEL
    # =========================
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_split=12,
        min_samples_leaf=6,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    # =========================
    # 7. SPLIT DATA
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================
    # 8. TRAIN
    # =========================
    pipeline.fit(X_train, y_train)

    # =========================
    # 9. BASE PREDICTIONS
    # =========================
    train_preds = pipeline.predict(X_train)
    test_preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    # =========================
    # 10. ACCURACY CHECK
    # =========================
    print("\n========================")
    print(f"Train Accuracy: {accuracy_score(y_train, train_preds):.4f}")
    print(f"Test Accuracy : {accuracy_score(y_test, test_preds):.4f}")
    print("========================")

    # =========================
    # 11. THRESHOLD ANALYSIS (KEY LEARNING)
    # =========================
    print("\n🔍 Threshold Analysis:")

    for t in [0.5, 0.4, 0.3, 0.2]:
        preds = (probs > t).astype(int)

        print(f"\nThreshold: {t}")
        print(confusion_matrix(y_test, preds))
        print(classification_report(y_test, preds))

    # =========================
    # 12. SAVE MODEL
    # =========================
    model_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../models")
    )
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "churn_pipeline.pkl")
    joblib.dump(pipeline, model_path)

    print("\n✅ Model saved successfully")

    return pipeline