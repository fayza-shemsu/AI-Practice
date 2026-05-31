import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

from src.data.load_data import load_data
from src.features.preprocess import build_preprocessor


def tune_random_forest(data_path):

    # =========================
    # 1. LOAD DATA
    # =========================
    df = load_data(data_path)
    print("✅ Data loaded successfully")

    # =========================
    # 2. CLEAN DATA
    # =========================
    if "CustomerID" in df.columns:
        df = df.drop("CustomerID", axis=1)

    # 🔥 REMOVE STRONG FEATURE (KEY CHANGE)
    if "Payment Delay" in df.columns:
        df = df.drop("Payment Delay", axis=1)
        print("🚫 Removed Payment Delay (to avoid trivial prediction)")

    df = df.drop_duplicates()

    # =========================
    # 3. SPLIT FEATURES / TARGET
    # =========================
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # =========================
    # 4. LEAKAGE CHECK
    # =========================
    print("\n🔍 Correlation Check with Target:")
    corr = df.corr(numeric_only=True)["Churn"].sort_values(ascending=False)
    print(corr)

    suspicious = corr[abs(corr) > 0.95].drop("Churn", errors="ignore")

    if len(suspicious) > 0:
        print("\n🚨 WARNING: HIGH LEAKAGE RISK FEATURES")
        print(suspicious)
    else:
        print("\n✔ No obvious leakage detected")

    # =========================
    # 5. PREPROCESSOR
    # =========================
    preprocessor = build_preprocessor(X)

    # =========================
    # 6. PIPELINE
    # =========================
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(random_state=42))
    ])

    # =========================
    # 7. PARAM GRID
    # =========================
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [8, 10, None],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2]
    }

    # =========================
    # 8. TRAIN / TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================
    # 9. GRID SEARCH
    # =========================
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    print("\n🔍 Running Grid Search...")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print("\n✅ Best Parameters:")
    print(grid.best_params_)

    # =========================
    # 10. TEST EVALUATION
    # =========================
    preds = best_model.predict(X_test)

    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\n📊 Classification Report:")
    print(classification_report(y_test, preds))

    test_f1 = f1_score(y_test, preds)
    print("\n🎯 Test F1 Score:", test_f1)

    # =========================
    # 11. CROSS-VALIDATION
    # =========================
    print("\n🔍 Cross-Validation (REAL PERFORMANCE):")

    cv_scores = cross_val_score(
        best_model,
        X,
        y,
        cv=5,
        scoring="f1"
    )

    print("CV Scores:", cv_scores)
    print("Mean CV F1:", cv_scores.mean())

    # =========================
    # 12. SHUFFLED TARGET TEST (FIXED)
    # =========================
    print("\n🔬 Leakage Robustness Test (Shuffled Target):")

    y_train_shuffled = y_train.sample(frac=1.0, random_state=42).reset_index(drop=True)
    X_train_reset = X_train.reset_index(drop=True)

    grid.fit(X_train_reset, y_train_shuffled)
    shuffled_score = grid.best_score_

    print("Shuffled CV F1:", shuffled_score)

    if shuffled_score > 0.6:
        print("🚨 WARNING: Possible leakage detected!")
    else:
        print("✔ Model is learning real patterns")

    # =========================
    # 13. OVERFITTING CHECK
    # =========================
    gap = test_f1 - cv_scores.mean()

    print("\n🧠 Model Stability Check:")
    print(f"Test F1: {test_f1:.4f}")
    print(f"CV Mean: {cv_scores.mean():.4f}")
    print(f"Gap     : {gap:.4f}")

    if gap > 0.10:
        print("\n⚠ WARNING: Possible overfitting risk")
    elif cv_scores.mean() > 0.95:
        print("\n⚠ WARNING: Suspiciously high performance")
    else:
        print("\n✔ Model performance looks realistic")

    # =========================
    # 14. SAVE MODEL
    # =========================
    model_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../models")
    )
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "tuned_random_forest.pkl")
    joblib.dump(best_model, model_path)

    print("\n💾 Model saved:", model_path)

    return best_model