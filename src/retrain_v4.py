import pandas as pd
import joblib
import os
import numpy as np
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

print("Loading data...")
creds = ml_client.datastores.get_default(include_secrets=True)
url   = (f"https://{creds.account_name}.blob.core.windows.net/"
         f"{creds.container_name}/"
         f"customer_churn_dataset-testing-master.csv"
         f"{creds.credentials.sas_token}")

df = pd.read_csv(url)
if "CustomerID" in df.columns:
    df = df.drop("CustomerID", axis=1)
leakage = ["Payment Delay", "Last Interaction"]
dropped = [c for c in leakage if c in df.columns]
if dropped:
    df = df.drop(dropped, axis=1)
    print("Dropped leakage:", dropped)
df = df.drop_duplicates()

TARGET = "Churn"
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df[TARGET]
)

print("\nChurn rate by gender in training data:")
print(train_df.groupby("Gender")["Churn"].mean().round(4))

# Keep gender for weight calculation then drop for training
train_gender = train_df["Gender"].copy()
test_gender  = test_df["Gender"].copy()

# Drop Gender from features — but use it to compute weights
X_train = train_df.drop([TARGET, "Gender"], axis=1)
y_train = train_df[TARGET]
X_test  = test_df.drop([TARGET, "Gender"], axis=1)
y_test  = test_df[TARGET]

X_train_enc = pd.get_dummies(X_train, drop_first=True)
X_test_enc  = pd.get_dummies(X_test,  drop_first=True)
X_test_enc  = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

# ── THE KEY FIX: sample weights per gender-churn group ───────
# We compute weights so that:
# Female churners    get higher weight (underrepresented in correct predictions)
# Female non-churners get normal weight
# Male churners      get higher weight
# Male non-churners  get normal weight
# This forces the model to learn each group's patterns equally

group_keys  = train_gender + "_" + y_train.astype(str)
group_counts = group_keys.value_counts()
total        = len(train_df)

sample_weights = group_keys.map(
    lambda g: total / (len(group_counts) * group_counts[g])
)

print("\nSample weight per group:")
for group, count in group_counts.items():
    weight = total / (len(group_counts) * count)
    print(f"  {group:20s}  count={count:6d}  weight={weight:.4f}")

# ── TRAIN VERSION 4 ───────────────────────────────────────────
print("\nTraining version 4 with group-aware sample weights...")
model_v4 = RandomForestClassifier(
    n_estimators  = 300,
    max_depth     = 15,
    min_samples_leaf = 5,
    random_state  = 42,
    n_jobs        = -1
)
model_v4.fit(X_train_enc, y_train, sample_weight=sample_weights.values)

preds = model_v4.predict(X_test_enc)
proba = model_v4.predict_proba(X_test_enc)[:, 1]

acc = accuracy_score(y_test, preds)
f1  = f1_score(y_test, preds)

print()
print("=" * 60)
print("VERSION 4 RESULTS — GROUP-AWARE WEIGHTS")
print("=" * 60)
print(f"Overall accuracy : {acc:.2%}")
print(f"Overall F1       : {f1:.4f}")
print(f"Errors           : {(preds != y_test).sum()} / {len(y_test)}")
print(f"Features         : {X_train_enc.columns.tolist()}")

print()
print("Accuracy by gender:")
for group in sorted(test_gender.unique()):
    mask   = test_gender == group
    g_acc  = accuracy_score(y_test[mask], preds[mask])
    g_fn   = ((preds[mask] == 0) & (y_test[mask].values == 1)).sum()
    g_churn = y_test[mask].sum()
    print(f"  {group:10s}  acc={g_acc:.2%}  "
          f"missed_churners={g_fn}/{g_churn}  "
          f"fnr={g_fn/g_churn:.2%}")

print()
print("Demographic parity (predicted churn rate by gender):")
for group in sorted(test_gender.unique()):
    mask = test_gender == group
    rate = preds[mask].mean()
    print(f"  {group:10s}  predicted churn rate: {rate:.2%}")

# Save model
os.makedirs("./rai_model_v4/", exist_ok=True)
model_path = "./rai_model_v4/churn_model_v4.pkl"
joblib.dump(model_v4, model_path)
print(f"\nModel saved: {model_path}")

# Register as version 4
registered = ml_client.models.create_or_update(
    Model(
        path        = model_path,
        name        = "churn-model-fayza",
        version     = "4",
        description = "Group-aware sample weights — gender-fair without using gender as feature",
        tags        = {
            "stage":           "staging",
            "algorithm":       "RandomForest",
            "gender_removed":  "true",
            "sample_weights":  "group_aware",
            "week":            "5",
            "owner":           "fayza"
        },
        type = AssetTypes.CUSTOM_MODEL
    )
)
print(f"Registered: {registered.name} version {registered.version}")
print()
print("Next: run fairness audit on version 4")