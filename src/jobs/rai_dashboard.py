import pandas as pd
import joblib
import os
import json
import shap
import numpy as np
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# ── LOAD MODEL v2 ─────────────────────────────────────────────
print("Loading model v2...")
os.makedirs("./rai_model_v2/", exist_ok=True)
ml_client.models.download(
    name="churn-model-fayza",
    version="2",
    download_path="./rai_model_v2/"
)
model_path = None
for root, dirs, files in os.walk("./rai_model_v2/"):
    for f in files:
        if f.endswith(".pkl"):
            model_path = os.path.join(root, f)
model = joblib.load(model_path)
print("Model loaded from:", model_path)
print("Model expects features:", model.feature_names_in_.tolist())

# ── LOAD DATA ─────────────────────────────────────────────────
print("\nLoading data...")
creds = ml_client.datastores.get_default(include_secrets=True)
url = f"https://{creds.account_name}.blob.core.windows.net/{creds.container_name}/customer_churn_dataset-testing-master.csv{creds.credentials.sas_token}"
df = pd.read_csv(url)
print("Raw data loaded:", df.shape)

# ── CLEAN — same as pipeline ──────────────────────────────────
if "CustomerID" in df.columns:
    df = df.drop("CustomerID", axis=1)

leakage = ["Payment Delay", "Last Interaction"]
dropped = [c for c in leakage if c in df.columns]
if dropped:
    df = df.drop(dropped, axis=1)
    print("Dropped leakage columns:", dropped)

df = df.drop_duplicates()
print("Cleaned data shape:", df.shape)
print("Cleaned columns:", df.columns.tolist())

# ── SPLIT ─────────────────────────────────────────────────────
TARGET = "Churn"
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df[TARGET]
)
print(f"Train: {len(train_df)}  Test: {len(test_df)}")

X_test   = test_df.drop(TARGET, axis=1)
y_test   = test_df[TARGET]
X_encoded = pd.get_dummies(X_test, drop_first=True)
X_encoded = X_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
preds     = model.predict(X_encoded)

# ── METRICS ───────────────────────────────────────────────────
acc = accuracy_score(y_test, preds)
f1  = f1_score(y_test, preds)

print()
print("=" * 60)
print("RAI RESULTS — churn-model-fayza v2")
print("=" * 60)
print(f"Accuracy : {acc:.2%}")
print(f"F1 Score : {f1:.4f}")
print(f"Errors   : {(preds != y_test).sum()} out of {len(y_test)}")

# ── ERROR CASES — save correctly ──────────────────────────────
# This is the fixed section. We attach predictions to test_df
# so the file always matches the current model and current run.
# error_cases.csv will only contain rows where the model was wrong.
os.makedirs("rai_output", exist_ok=True)

results_df          = test_df.copy()
results_df          = results_df.reset_index(drop=True)
results_df["predicted"] = preds
results_df["correct"]   = (preds == y_test.values)

error_cases = results_df[results_df["correct"] == False].copy()
error_cases.to_csv("rai_output/error_cases.csv", index=True)

print()
print("ERROR CASES")
print("-" * 60)
print(f"Total wrong predictions : {len(error_cases)}")
print(f"Total test customers    : {len(test_df)}")
print(f"Calculated accuracy     : {(len(test_df) - len(error_cases)) / len(test_df) * 100:.2f}%")
print(f"Saved to                : rai_output/error_cases.csv")

# verify the file matches the accuracy number
file_df   = pd.read_csv("rai_output/error_cases.csv")
file_acc  = (len(test_df) - len(file_df)) / len(test_df) * 100
match     = abs(file_acc - acc * 100) < 0.01
print(f"File verification       : {'MATCH ✅' if match else 'MISMATCH ❌'}")

# ── ERROR ANALYSIS BY GROUP ───────────────────────────────────
print()
print("ERROR ANALYSIS BY GROUP")
print("-" * 60)

categorical_cols = [
    c for c in test_df.columns
    if c != TARGET and test_df[c].dtype == object
]

error_analysis = {}
for col in categorical_cols:
    print(f"\nBy {col}:")
    col_results = {}
    for group in sorted(test_df[col].unique()):
        mask = test_df[col] == group
        if mask.sum() < 5:
            continue
        g_preds  = model.predict(X_encoded[mask.values])
        g_true   = y_test[mask.values]
        g_acc    = accuracy_score(g_true, g_preds)
        errors   = (g_preds != g_true).sum()
        total    = mask.sum()

        # false positives: predicted churn=1 but actually stayed=0
        fp = ((g_preds == 1) & (g_true.values == 0)).sum()
        # false negatives: predicted stay=0 but actually churned=1
        fn = ((g_preds == 0) & (g_true.values == 1)).sum()

        print(f"  {str(group):25s} acc={g_acc:.2%}  "
              f"errors={errors}/{total}  "
              f"missed_churners(FN)={fn}  "
              f"wrong_flags(FP)={fp}")

        col_results[str(group)] = {
            "accuracy":         round(g_acc, 4),
            "errors":           int(errors),
            "total":            int(total),
            "false_negatives":  int(fn),
            "false_positives":  int(fp),
        }
    error_analysis[col] = col_results

# ── SHAP FEATURE IMPORTANCE ───────────────────────────────────
print()
print("FEATURE IMPORTANCE (SHAP)")
print("-" * 60)
print("Computing SHAP values on 500 sample rows...")

sample      = X_encoded.sample(min(500, len(X_encoded)), random_state=42)
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

mean_shap = pd.Series(
    abs(shap_values[1]).mean(axis=0),
    index=model.feature_names_in_
).sort_values(ascending=False)

print("Top 10 features:")
for i, (feat, val) in enumerate(mean_shap.head(10).items()):
    print(f"  {i+1:2d}. {feat:35s} {val:.4f}")

mean_shap.to_csv("rai_output/feature_importance.csv")

# ── FAIRNESS ANALYSIS ─────────────────────────────────────────
print()
print("FAIRNESS ANALYSIS")
print("-" * 60)

fairness = {}
for col in categorical_cols:
    accs         = []
    group_results = {}

    for group in test_df[col].unique():
        mask = test_df[col] == group
        if mask.sum() < 10:
            continue
        g_preds = model.predict(X_encoded[mask.values])
        g_acc   = accuracy_score(y_test[mask.values], g_preds)
        accs.append(g_acc)
        group_results[str(group)] = round(g_acc, 4)

    if len(accs) >= 2:
        gap    = max(accs) - min(accs)
        status = (
            "FAIR"    if gap < 0.05 else
            "WARNING" if gap < 0.10 else
            "BIASED"
        )
        print(f"\n{col} — gap={gap:.2%} [{status}]:")
        for group, g_acc in group_results.items():
            flag = " ← lowest" if g_acc == min(accs) else ""
            print(f"  {str(group):25s} -> {g_acc:.2%}{flag}")

        fairness[col] = {
            "gap":     round(gap, 4),
            "status":  status,
            "groups":  group_results
        }

# ── SAVE RAI SUMMARY ──────────────────────────────────────────
rai_results = {
    "model":            "churn-model-fayza v2",
    "accuracy":         round(acc, 4),
    "f1_score":         round(f1, 4),
    "total_test":       len(test_df),
    "total_errors":     int((preds != y_test).sum()),
    "leakage_removed":  ["Payment Delay", "Last Interaction"],
    "error_analysis":   error_analysis,
    "fairness":         fairness,
    "top_features":     mean_shap.head(10).round(4).to_dict(),
}

with open("rai_output/rai_summary.json", "w") as f:
    json.dump(rai_results, f, indent=2)

# ── FINAL SUMMARY ─────────────────────────────────────────────
print()
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Model           : churn-model-fayza v2")
print(f"Accuracy        : {acc:.2%}")
print(f"F1 Score        : {f1:.4f}")
print(f"Total errors    : {(preds != y_test).sum()} / {len(test_df)}")
print(f"Leakage removed : {dropped}")
print()
print("Files saved:")
print("  rai_output/error_cases.csv        ← wrong predictions only")
print("  rai_output/feature_importance.csv ← SHAP ranking")
print("  rai_output/rai_summary.json       ← full audit report")
print()
print("Saved to rai_output/")
print("Tuesday Week 5 COMPLETE")