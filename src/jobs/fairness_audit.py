import pandas as pd
import joblib
import os
import json
import sys
import numpy as np
from datetime import datetime
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    confusion_matrix
)

try:
    from fairlearn.metrics import (
        MetricFrame,
        selection_rate,
        true_positive_rate,
        false_positive_rate,
        false_negative_rate,
    )
    FAIRLEARN_AVAILABLE = True
except ImportError:
    print("fairlearn not installed — running manual fairness audit")
    FAIRLEARN_AVAILABLE = False

# ── CONNECT AND LOAD ──────────────────────────────────────────
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

print("Loading model v2...")
os.makedirs("./rai_model_v2/", exist_ok=True)
ml_client.models.download(
    name="churn-model-fayza",
    version="2",
    download_path="./rai_model_v2/"
)
model_path = None
for root, dirs, files in os.walk("./rai_model_v4/"):
    for f in files:
        if f.endswith(".pkl"):
            model_path = os.path.join(root, f)
            break

model = joblib.load(model_path)
print("Model loaded:", type(model).__name__)
print("Model expects features:", model.feature_names_in_.tolist())

print("Loading registered data: churn-dataset@latest")
creds     = ml_client.datastores.get_default(include_secrets=True)
account   = creds.account_name
container = creds.container_name
sas_token = creds.credentials.sas_token
blob_file = "customer_churn_dataset-testing-master.csv"
url       = (f"https://{account}.blob.core.windows.net/"
             f"{container}/{blob_file}{sas_token}")

df = pd.read_csv(url)
print("Raw data loaded:", df.shape)

if "CustomerID" in df.columns:
    df = df.drop("CustomerID", axis=1)

# FIX 2 — use dropped variable, not hardcoded list
leakage = ["Payment Delay", "Last Interaction"]
dropped = [c for c in leakage if c in df.columns]
if dropped:
    df = df.drop(dropped, axis=1)
    print("Dropped leakage columns:", dropped)

df = df.drop_duplicates()
print("Cleaned data shape:", df.shape)

TARGET = "Churn"
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df[TARGET]
)

sensitive_features = test_df[["Gender"]].copy()
X_test    = test_df.drop(TARGET, axis=1)
y_test    = test_df[TARGET]
X_encoded = pd.get_dummies(X_test, drop_first=True)
X_encoded = X_encoded.fillna(0)
X_encoded = X_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

preds = model.predict(X_encoded)
proba = model.predict_proba(X_encoded)[:, 1]

# FIX 1 — dynamic audit date
audit_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print()
print("=" * 60)
print("FAIRNESS AUDIT — churn-model-fayza v2")
print(f"Sensitive feature: Gender")
print(f"Audit date: {audit_timestamp}")
print("=" * 60)

overall_acc = accuracy_score(y_test, preds)
overall_f1  = f1_score(y_test, preds)
print(f"\nOverall accuracy: {overall_acc:.2%}")
print(f"Overall F1:       {overall_f1:.4f}")
print(f"Total test rows:  {len(y_test)}")

groups = test_df["Gender"].unique()

# ── CONCEPT 1: DEMOGRAPHIC PARITY ────────────────────────────
print()
print("CONCEPT 1 — DEMOGRAPHIC PARITY")
print("Does the model predict churn equally across genders?")
print("(regardless of whether predictions are correct)")
print("-" * 60)

pred_rates  = {}
n_per_group = {}
for group in sorted(groups):
    mask      = test_df["Gender"] == group
    pred_rate = preds[mask].mean()
    pred_rates[group]  = pred_rate
    n_per_group[group] = int(mask.sum())
    print(f"  {group:10s}  predicted churn rate: {pred_rate:.2%}  (n={mask.sum()})")

dp_gap    = max(pred_rates.values()) - min(pred_rates.values())
dp_status = (
    "FAIR ✅"    if dp_gap < 0.05 else
    "WARNING ⚠️" if dp_gap < 0.10 else
    "BIASED ❌"
)
print(f"\n  Demographic parity gap: {dp_gap:.2%}  [{dp_status}]")
print(f"  Threshold: < 5% = fair,  5-10% = warning,  > 10% = biased")
print()
print("  WHAT THIS MEANS:")
print(f"  The model flags {max(pred_rates.values()):.2%} of "
      f"{max(pred_rates, key=pred_rates.get)} customers as churners")
print(f"  but only {min(pred_rates.values()):.2%} of "
      f"{min(pred_rates, key=pred_rates.get)} customers.")
print(f"  That is a {dp_gap:.2%} difference in how often each group gets flagged.")

# ── CONCEPT 2: EQUAL OPPORTUNITY ─────────────────────────────
print()
print("CONCEPT 2 — EQUAL OPPORTUNITY")
print("Among customers who ACTUALLY churn, does the model")
print("correctly identify them equally across genders?")
print("-" * 60)

tpr_rates = {}
fnr_rates = {}
tp_counts = {}
fn_counts = {}
for group in sorted(groups):
    mask   = test_df["Gender"] == group
    g_true = y_test[mask]
    g_pred = preds[mask]
    tn, fp, fn, tp = confusion_matrix(g_true, g_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
    tpr_rates[group] = tpr
    fnr_rates[group] = fnr
    tp_counts[group] = int(tp)
    fn_counts[group] = int(fn)
    print(f"  {group:10s}")
    print(f"    Caught churners (TPR) : {tpr:.2%}  "
          f"({tp} caught out of {tp+fn} actual churners)")
    print(f"    Missed churners (FNR) : {fnr:.2%}  "
          f"({fn} missed — these customers left without a call)")

eo_gap    = max(tpr_rates.values()) - min(tpr_rates.values())
eo_status = (
    "FAIR ✅"    if eo_gap < 0.05 else
    "WARNING ⚠️" if eo_gap < 0.10 else
    "BIASED ❌"
)
print(f"\n  Equal opportunity gap: {eo_gap:.2%}  [{eo_status}]")

worst_group = min(tpr_rates, key=tpr_rates.get)
best_group  = max(tpr_rates, key=tpr_rates.get)
ratio       = fnr_rates[worst_group] / fnr_rates[best_group] if fnr_rates[best_group] > 0 else 0
print(f"\n  WHAT THIS MEANS:")
print(f"  The model misses {fnr_rates[worst_group]:.2%} of {worst_group} churners")
print(f"  but only {fnr_rates[best_group]:.2%} of {best_group} churners.")
print(f"  It is {ratio:.1f}x more likely to miss a {worst_group} churner.")
print(f"  Every missed churner = a customer who left without a retention call.")

# ── CONCEPT 3: ACCURACY PER GROUP ────────────────────────────
print()
print("CONCEPT 3 — ACCURACY PER GROUP")
print("Does the model perform equally well for each gender?")
print("-" * 60)

acc_rates = {}
f1_rates  = {}
for group in sorted(groups):
    mask   = test_df["Gender"] == group
    g_true = y_test[mask]
    g_pred = preds[mask]
    acc    = accuracy_score(g_true, g_pred)
    f1     = f1_score(g_true, g_pred)
    acc_rates[group] = acc
    f1_rates[group]  = f1
    errors = (g_pred != g_true.values).sum()
    print(f"  {group:10s}  accuracy={acc:.2%}  "
          f"f1={f1:.4f}  errors={errors}/{mask.sum()}  n={mask.sum()}")

acc_gap    = max(acc_rates.values()) - min(acc_rates.values())
acc_status = (
    "FAIR ✅"    if acc_gap < 0.05 else
    "WARNING ⚠️" if acc_gap < 0.10 else
    "BIASED ❌"
)
print(f"\n  Accuracy gap: {acc_gap:.2%}  [{acc_status}]")
print(f"\n  WHAT THIS MEANS:")
print(f"  For every 100 {best_group} customers the model gets "
      f"{max(acc_rates.values())*100:.0f} right.")
print(f"  For every 100 {worst_group} customers it gets only "
      f"{min(acc_rates.values())*100:.0f} right.")

# ── CONCEPT 4: CALIBRATION ────────────────────────────────────
print()
print("CONCEPT 4 — CALIBRATION")
print("When model says 80% churn probability, is it right 80%")
print("of the time — equally across genders?")
print("-" * 60)

bins   = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]

calibration_summary = {}
for group in sorted(groups):
    mask      = test_df["Gender"] == group
    g_proba   = proba[mask]
    g_true    = y_test[mask].values
    g_bin     = pd.cut(g_proba, bins=bins, labels=labels)
    group_cal = {}

    print(f"\n  {group}:")
    print(f"  {'Bucket':12s}  {'Predicted':10s}  {'Actual':10s}  {'OK?':6s}  {'n':>5}")
    print(f"  {'-'*50}")
    for label in labels:
        bucket_mask   = g_bin == label
        if bucket_mask.sum() < 5:
            continue
        actual_rate   = g_true[bucket_mask].mean()
        bucket_center = bins[labels.index(label)] + 0.1
        calibrated    = abs(actual_rate - bucket_center) < 0.15
        ok            = "✅" if calibrated else "❌"
        print(f"  {label:12s}  {bucket_center:.0%}            "
              f"{actual_rate:.0%}          {ok}      {bucket_mask.sum()}")
        group_cal[label] = {
            "predicted":  round(bucket_center, 2),
            "actual":     round(float(actual_rate), 4),
            "n":          int(bucket_mask.sum()),
            "calibrated": bool(calibrated)
        }
    calibration_summary[group] = group_cal

# ── FAIRLEARN ─────────────────────────────────────────────────
if FAIRLEARN_AVAILABLE:
    print()
    print("FAIRLEARN — STANDARDIZED FAIRNESS METRICS")
    print("-" * 60)
    metrics = {
        "accuracy":            accuracy_score,
        "f1":                  f1_score,
        "selection_rate":      selection_rate,
        "true_positive_rate":  true_positive_rate,
        "false_negative_rate": false_negative_rate,
    }
    mf = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=preds,
        sensitive_features=sensitive_features["Gender"]
    )
    print("\nMetrics by group:")
    print(mf.by_group.to_string())
    print("\nOverall:")
    print(mf.overall.to_string())
    print("\nDifference (max gap between groups):")
    print(mf.difference().to_string())

# ── AUDIT VERDICT ─────────────────────────────────────────────
print()
print("=" * 60)
print("AUDIT VERDICT")
print("=" * 60)

issues = []
if dp_gap  > 0.05:
    issues.append(
        f"Demographic parity gap {dp_gap:.2%} exceeds 5% threshold")
if eo_gap  > 0.05:
    issues.append(
        f"Equal opportunity gap {eo_gap:.2%} exceeds 5% threshold")
if acc_gap > 0.05:
    issues.append(
        f"Accuracy gap {acc_gap:.2%} exceeds 5% threshold")

if issues:
    verdict = "BLOCKED — FAIRNESS ISSUES FOUND ❌"
else:
    verdict = "APPROVED FOR DEPLOYMENT ✅"

print(f"\nVerdict: {verdict}")

if issues:
    print("\nIssues found:")
    for issue in issues:
        print(f"  - {issue}")
    print()
    print("Action required before deployment:")
    print("  1. Retrain model without Gender as a feature (version 3)")
    print("  2. Re-run this fairness audit on version 3")
    print("  3. All gaps must be below 5% threshold to proceed")
    print("  4. Document remediation steps for compliance record")
else:
    print("All fairness metrics pass the 5% threshold.")
    print("Model is cleared for production deployment.")

# ── SAVE AUDIT REPORT ─────────────────────────────────────────
os.makedirs("./rai_output/fairness", exist_ok=True)

audit_report = {
    "model":             "churn-model-fayza v2",
    "registered_data":   "churn-dataset@latest",
    "sensitive_feature": "Gender",
    "leakage_removed":   dropped,           # FIX 2 — dynamic
    "audit_date":        audit_timestamp,   # FIX 1 — dynamic
    "verdict":           verdict,
    "overall": {
        "accuracy": round(overall_acc, 4),
        "f1":       round(overall_f1, 4),
        "n":        len(y_test),
    },
    "metrics": {
        "demographic_parity_gap": round(dp_gap, 4),
        "equal_opportunity_gap":  round(eo_gap, 4),
        "accuracy_gap":           round(acc_gap, 4),
        "dp_status":              dp_status,
        "eo_status":              eo_status,
        "acc_status":             acc_status,
    },
    "by_group": {                           # ADDITION 5 — n per group
        group: {
            "n":                    n_per_group[group],
            "predicted_churn_rate": round(pred_rates[group], 4),
            "true_positive_rate":   round(tpr_rates[group], 4),
            "false_negative_rate":  round(fnr_rates[group], 4),
            "caught_churners":      tp_counts[group],
            "missed_churners":      fn_counts[group],
            "accuracy":             round(acc_rates[group], 4),
            "f1":                   round(f1_rates[group], 4),
        }
        for group in sorted(groups)
    },
    "calibration":  calibration_summary,   # ADDITION 4
    "issues":       issues,
    "recommended_actions": [
        "Retrain churn-model-fayza version 3 without Gender feature",
        "Re-run fairness audit on version 3",
        "All demographic gaps must be below 5% threshold",
        "Document remediation steps for compliance record"
    ]
}

with open("./rai_output/fairness/audit_report.json", "w") as f:
    json.dump(audit_report, f, indent=2)

print()
print("Saved: ./rai_output/fairness/audit_report.json")
print()
print("Friday Week 5 COMPLETE")
print()

# FIX 3 — exit code 1 stops CI/CD pipeline when blocked
if issues:
    print("Exit code 1 — deployment pipeline is now blocked.")
    sys.exit(1)
else:
    sys.exit(0)