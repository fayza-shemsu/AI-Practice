import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.data.load_data import load_data


def run_classification_pipeline(data_path):

    # 1. LOAD DATA
    df = load_data(data_path)
    print(df["Churn"].unique())
    df = df.drop("CustomerID", axis=1)
    print(df.isnull().sum())
    print(df.duplicated().sum())

    # 3. SPLIT FEATURES / TARGET
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # 4. IDENTIFY COLUMN TYPES
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    # 5. PREPROCESSING PIPELINE
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop='first'), cat_cols)
        ]
    )

    # 6. FULL PIPELINE
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
("model", RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
))    ])

    # 7. TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 8. TRAIN
    pipeline.fit(X_train, y_train)
    
        # 9. PREDICT WITH PROBABILITY
    probs = pipeline.predict_proba(X_test)[:, 1]

    # LOWER THRESHOLD TO REDUCE FN
    threshold = 0.3
    preds = (probs > threshold).astype(int)
    
     # 10. OVERFITTING CHECK
    train_preds = pipeline.predict(X_train)
    test_preds = pipeline.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    print(f"Train Accuracy: {train_acc}")
    print(f"Test Accuracy: {test_acc}")

    # 11. EVALUATE
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # 12. SAVE MODEL
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "churn_pipeline.pkl")
    joblib.dump(pipeline, model_path)

    return pipeline