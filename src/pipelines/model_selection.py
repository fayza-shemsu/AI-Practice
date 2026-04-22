import os
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.data.load_data import load_data


def run_model_selection(data_path):

    # 1. LOAD DATA
    df = load_data(data_path)
    print("✅ Data loaded successfully")

    # DROP ID IF EXISTS
    if "CustomerID" in df.columns:
        df = df.drop("CustomerID", axis=1)

    # 2. SPLIT
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # 3. COLUMN TYPES
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # 4. PREPROCESSOR
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first"), cat_cols)
    ])

    # 5. MODELS
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC()
    }

    # 6. SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # 7. TRAIN MODELS
    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        # predictions
        train_preds = pipeline.predict(X_train)
        test_preds = pipeline.predict(X_test)

        # metrics
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds)

        results[name] = {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "f1": test_f1
        }

        print(f"\n{name}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy : {test_acc:.4f}")
        print(f"F1 Score      : {test_f1:.4f}")

        # 🔍 CONFUSION MATRIX
        cm = confusion_matrix(y_test, test_preds)
        print("Confusion Matrix:")
        print(cm)

    # 8. BEST MODEL
    best_model = max(results, key=lambda x: results[x]["f1"])
    best_score = results[best_model]["f1"]

    print("\n🏆 Leaderboard (by F1-score):")
    for k, v in sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True):
        print(f"{k}: F1={v['f1']:.4f}, Train={v['train_acc']:.4f}, Test={v['test_acc']:.4f}")

    print(f"\n✅ Best Model: {best_model} with F1 = {best_score:.4f}")

    # 9. 🔍 CROSS VALIDATION CHECK (STABILITY)
    print("\n🔍 Cross-Validation Check (RandomForest):")

    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    cv_scores = cross_val_score(
        rf,
        X,
        y,
        cv=5,
        scoring="f1"
    )

    print(cv_scores)
    print("Mean F1:", cv_scores.mean())

    # 10. 🔍 FEATURE IMPORTANCE CHECK (LEAKAGE DETECTION)
    print("\n📊 Feature Importance Check:")

    rf.fit(X_train, y_train)

    importances = pd.Series(rf.feature_importances_, index=X.columns)
    print(importances.sort_values(ascending=False))

    # 11. 🧠 MODEL HEALTH CHECK
    print("\n🧠 MODEL HEALTH CHECK")

    if best_score > 0.95:
        print("⚠ WARNING: Very high score → possible easy dataset or leakage risk")
    else:
        print("✔ Score looks realistic")

    return results