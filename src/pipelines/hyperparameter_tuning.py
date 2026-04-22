import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier

from src.data.load_data import load_data


def tune_random_forest(data_path):

    # 1. LOAD DATA
    df = load_data(data_path)

    # 🚨 REMOVE LEAKAGE
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

    # 5. PIPELINE
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(random_state=42))
    ])

    # 6. PARAM GRID (THIS IS THE CORE)
    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2]
    }

    # 7. SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 8. GRID SEARCH
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    print("🔍 Running Grid Search...")
    grid.fit(X_train, y_train)

    # 9. BEST MODEL
    best_model = grid.best_estimator_

    print("\n✅ Best Parameters:")
    print(grid.best_params_)

    # 10. EVALUATION
    preds = best_model.predict(X_test)

    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\n📊 Classification Report:")
    print(classification_report(y_test, preds))

    f1 = f1_score(y_test, preds)
    print("\n🎯 Final F1 Score:", f1)

    # 11. SAVE MODEL
    model_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../models")
    )
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "tuned_random_forest.pkl")
    joblib.dump(best_model, model_path)

    print("\n💾 Model saved:", model_path)

    return best_model