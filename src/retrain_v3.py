import pandas as pd
import joblib
import os
import json
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

print("Loading data...")
creds     = ml_client.datastores.get_default(include_secrets=True)
url       = (f"https://{creds.account_name}.blob.core.windows.net/"
             f"{creds.container_name}/"
             f"customer_churn_dataset-testing-master.csv"
             f"{creds.credentials.sas_token}")

df = pd.read_csv(url)

# Drop leakage columns
if "CustomerID" in df.columns:
    df = df.drop("CustomerID", axis=1)
leakage = ["Payment Delay", "Last Interaction"]
dropped = [c for c in leakage if c in df.columns]
if dropped:
    df = df.drop(dropped, axis=1)
    print("Dropped leakage:", dropped)

# DROP GENDER — this is the fix
if "Gender" in df.columns:
    df = df.drop("Gender", axis=1)
    print("Dropped Gender column — removing bias source")

df = df.drop_duplicates()
print("Data shape after all drops:", df.shape)
print("Remaining columns:", df.columns.tolist())

TARGET = "Churn"
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df[TARGET]
)

X_train = train_df.drop(TARGET, axis=1)
y_train = train_df[TARGET]
X_test  = test_df.drop(TARGET, axis=1)
y_test  = test_df[TARGET]

X_train_enc = pd.get_dummies(X_train, drop_first=True)
X_test_enc  = pd.get_dummies(X_test,  drop_first=True)
X_test_enc  = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

print("\nTraining version 3 WITHOUT Gender...")
model_v3 = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",   # fixes class imbalance too
    random_state=42,
    n_jobs=-1
)
model_v3.fit(X_train_enc, y_train)

preds = model_v3.predict(X_test_enc)
acc   = accuracy_score(y_test, preds)
f1    = f1_score(y_test, preds)

print()
print("=" * 60)
print("VERSION 3 RESULTS")
print("=" * 60)
print(f"Accuracy : {acc:.2%}")
print(f"F1 Score : {f1:.4f}")
print(f"Errors   : {(preds != y_test).sum()} out of {len(y_test)}")
print(f"Features : {X_train_enc.columns.tolist()}")

# Save model
os.makedirs("./rai_model_v3/", exist_ok=True)
model_path = "./rai_model_v3/churn_model_v3.pkl"
joblib.dump(model_v3, model_path)
print(f"\nModel saved: {model_path}")

# Register as version 3 in Azure ML
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

registered = ml_client.models.create_or_update(
    Model(
        path        = model_path,
        name        = "churn-model-fayza",
        version     = "3",
        description = "Retrained without Gender feature to fix fairness BLOCKED verdict",
        tags        = {
            "stage":          "staging",
            "algorithm":      "RandomForest",
            "gender_removed": "true",
            "class_weight":   "balanced",
            "week":           "5",
            "owner":          "fayza"
        },
        type = AssetTypes.CUSTOM_MODEL
    )
)
print(f"Registered: {registered.name} version {registered.version}")
print()
print("Next step: run fairness_audit.py pointing at version 3")
print("Expected: demographic parity gap should drop from 16% to near 0%")