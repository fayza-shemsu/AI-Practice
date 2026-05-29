import json
import os
import joblib
import pandas as pd

def init():
    global model

    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    print("Model dir:", model_dir)
    print("Contents:", os.listdir(model_dir))

    # Search recursively for model.pkl
    model_path = None
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".pkl"):
                model_path = os.path.join(root, f)
                break
        if model_path:
            break

    if model_path is None:
        raise FileNotFoundError("model.pkl not found in " + model_dir)

    print("Loading from:", model_path)
    model = joblib.load(model_path)
    print("Model loaded:", type(model).__name__)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        if isinstance(data, dict):
            data = [data]
        df = pd.DataFrame(data)
        encoded = pd.get_dummies(df, drop_first=True)
        encoded = encoded.reindex(columns=model.feature_names_in_, fill_value=0)
        predictions   = model.predict(encoded)
        probabilities = model.predict_proba(encoded)[:, 1]
        results = []
        for pred, prob in zip(predictions, probabilities):
            risk = "HIGH" if prob >= 0.8 else "MEDIUM" if prob >= 0.5 else "LOW"
            results.append({
                "churn":             bool(pred),
                "churn_probability": round(float(prob), 4),
                "risk_level":        risk
            })
        return json.dumps(results)
    except Exception as e:
        print("ERROR:", str(e))
        return json.dumps({"error": str(e)})
