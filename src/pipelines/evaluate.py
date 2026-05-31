import pandas as pd
import joblib
import argparse
import os
from sklearn.metrics import f1_score, accuracy_score

def main(model_path, data_path):

    # FIX 1: if Azure gives a folder, find the .pkl file inside it
    if os.path.isdir(model_path):
        pkl_files = [f for f in os.listdir(model_path) if f.endswith(".pkl")]
        model_path = os.path.join(model_path, pkl_files[0])

    # FIX 2: if Azure gives a folder, find the .csv file inside it
    if os.path.isdir(data_path):
        csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
        data_path = os.path.join(data_path, csv_files[0])

    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # FIX 3: encode text columns exactly like train.py did
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)

    # FIX 4: reindex columns to match exactly what the model was trained on
    X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    preds = model.predict(X)

    print("🎯 F1 Score:", f1_score(y, preds))
    print("✅ Accuracy:", accuracy_score(y, preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()
    main(args.model_path, args.data_path)