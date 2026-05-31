import pandas as pd
import argparse
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main(input_data, model_output):
    df = pd.read_csv(input_data)
    print("Data shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Churn value counts:\n", df["Churn"].value_counts())

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Convert ALL text columns to numbers automatically
    X = pd.get_dummies(X, drop_first=True)
    print("After encoding, shape:", X.shape)
    print("Any nulls?", X.isnull().sum().sum())

    # Drop any remaining nulls just in case
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # model_output is a file path like /outputs/model.pkl
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump(model, model_output)
    print("Model saved at:", model_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--model_output", type=str)
    args = parser.parse_args()
    main(args.input_data, args.model_output)
