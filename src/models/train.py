import pandas as pd
import argparse
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main(input_data, model_output):
    df = pd.read_csv(input_data)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # encode categorical columns
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump(model, model_output)
    print("✅ Model saved at:", model_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--model_output", type=str)
    args = parser.parse_args()
    main(args.input_data, args.model_output)