import pandas as pd
import joblib
import os
import json
import shap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── CONNECT AND LOAD ──────────────────────────────────────────
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

print("Loading model...")
model_path = None
for root, dirs, files in os.walk("./rai_model_v2/"):
    for f in files:
        if f.endswith(".pkl"):
            model_path = os.path.join(root, f)
            break

if model_path is None:
    ml_client.models.download(
        name="churn-model-fayza",
        version="2",
        download_path="./rai_model_v2/"
    )
    for root, dirs, files in os.walk("./rai_model_v2/"):
        for f in files:
            if f.endswith(".pkl"):
                model_path = os.path.join(root, f)
                break

model = joblib.load(model_path)
print("Model loaded:", type(model).__name__)
print("Features:", model.feature_names_in_.tolist())

# ── LOAD AND PREPARE DATA ─────────────────────────────────────
print("\nLoading data...")
creds     = ml_client.datastores.get_default(include_secrets=True)
url       = (f"https://{creds.account_name}.blob.core.windows.net/"
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

X_test    = test_df.drop(TARGET, axis=1)
y_test    = test_df[TARGET]
X_encoded = pd.get_dummies(X_test, drop_first=True)
X_encoded = X_encoded.fillna(0)
X_encoded = X_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
preds     = model.predict(X_encoded)
probs     = model.predict_proba(X_encoded)[:, 1]

print("Data ready:", X_encoded.shape)

os.makedirs("./rai_output/shap", exist_ok=True)

# ── COMPUTE SHAP ──────────────────────────────────────────────
print("\nComputing SHAP values (500 sample rows)...")
sample    = X_encoded.sample(min(500, len(X_encoded)), random_state=42)
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(sample)

# shap_vals[1] = contributions toward churn prediction
# Positive value = pushes TOWARD churn
# Negative value = pushes AWAY from churn
shap_churn = shap_vals[1]

# ── LEVEL 1: GLOBAL FEATURE IMPORTANCE ───────────────────────
print("\n" + "=" * 60)
print("LEVEL 1 — GLOBAL FEATURE IMPORTANCE")
print("Which features matter most across ALL customers?")
print("=" * 60)

mean_shap = pd.Series(
    abs(shap_churn).mean(axis=0),
    index=model.feature_names_in_
).sort_values(ascending=False)

print("\nFeature importance ranking:")
for i, (feature, importance) in enumerate(mean_shap.items()):
    bar = "█" * int(importance * 200)
    print(f"  {i+1:2d}. {feature:35s} {importance:.4f}  {bar}")

plt.figure(figsize=(10, 6))
mean_shap.sort_values().plot(kind="barh", color="steelblue")
plt.title("Global Feature Importance (mean |SHAP|)\nchurn-model-fayza v2")
plt.xlabel("Mean |SHAP value| — average impact on churn prediction")
plt.tight_layout()
plt.savefig("./rai_output/shap/global_importance.png", dpi=150)
plt.close()
print("\nSaved: global_importance.png")

# ── LEVEL 2: DIRECTIONAL SHAP — THE COMPLETE VERSION ─────────
print("\n" + "=" * 60)
print("LEVEL 2 — DIRECTIONAL SHAP ANALYSIS")
print("For each feature: does HIGH value increase or decrease churn?")
print("=" * 60)

print()
print(f"  {'Feature':<35} {'Direction':<40} {'Avg SHAP when HIGH':>20}")
print(f"  {'-'*100}")

directional_results = {}
for feature in mean_shap.index:
    col_idx      = list(model.feature_names_in_).index(feature)
    feat_vals    = sample[feature].values
    feat_shap    = shap_churn[:, col_idx]

    # Split customers into HIGH and LOW value groups
    # HIGH = above median, LOW = at or below median
    median_val   = np.median(feat_vals)
    high_mask    = feat_vals > median_val
    low_mask     = feat_vals <= median_val

    avg_shap_high = feat_shap[high_mask].mean() if high_mask.sum() > 0 else 0
    avg_shap_low  = feat_shap[low_mask].mean()  if low_mask.sum()  > 0 else 0

    # Correlation gives overall direction
    if len(np.unique(feat_vals)) > 1:
        correlation = np.corrcoef(feat_vals, feat_shap)[0, 1]
    else:
        correlation = 0

    # Determine direction from SHAP values directly
    if avg_shap_high > 0.005:
        direction     = "HIGH value → MORE churn risk   ↑"
        direction_key = "increases_churn"
        action        = f"Reduce {feature}"
    elif avg_shap_high < -0.005:
        direction     = "HIGH value → LESS churn risk   ↓"
        direction_key = "decreases_churn"
        action        = f"Increase {feature}"
    else:
        direction     = "Mixed effect — depends on customer"
        direction_key = "mixed"
        action        = f"Investigate {feature} per segment"

    print(f"  {feature:<35} {direction:<40} {avg_shap_high:>+.4f}")

    directional_results[feature] = {
        "global_importance":  round(float(mean_shap[feature]), 4),
        "avg_shap_high_val":  round(float(avg_shap_high), 4),
        "avg_shap_low_val":   round(float(avg_shap_low), 4),
        "direction":          direction_key,
        "correlation":        round(float(correlation), 4),
        "recommended_action": action,
    }

# ── LEVEL 3: DIRECTIONAL SUMMARY TABLE ───────────────────────
print("\n" + "=" * 60)
print("LEVEL 3 — DIRECTIONAL SUMMARY TABLE")
print("Production-ready interpretation for business teams")
print("=" * 60)

increases = [(f, d) for f, d in directional_results.items()
             if d["direction"] == "increases_churn"]
decreases = [(f, d) for f, d in directional_results.items()
             if d["direction"] == "decreases_churn"]
mixed     = [(f, d) for f, d in directional_results.items()
             if d["direction"] == "mixed"]

increases.sort(key=lambda x: abs(x[1]["global_importance"]), reverse=True)
decreases.sort(key=lambda x: abs(x[1]["global_importance"]), reverse=True)

print("\n  RISK FACTORS — high value increases churn probability:")
for feature, d in increases:
    print(f"    {feature:<35} importance={d['global_importance']:.4f}  "
          f"action: {d['recommended_action']}")

print("\n  PROTECTIVE FACTORS — high value decreases churn probability:")
for feature, d in decreases:
    print(f"    {feature:<35} importance={d['global_importance']:.4f}  "
          f"action: {d['recommended_action']}")

if mixed:
    print("\n  MIXED FACTORS — effect depends on customer segment:")
    for feature, d in mixed:
        print(f"    {feature:<35} importance={d['global_importance']:.4f}")

# ── LEVEL 4: INDIVIDUAL CUSTOMER EXPLANATION ─────────────────
print("\n" + "=" * 60)
print("LEVEL 4 — INDIVIDUAL CUSTOMER EXPLANATION")
print("Why is THIS specific customer predicted to churn?")
print("=" * 60)

# Find highest risk customer
top_idx      = probs.argmax()
customer     = X_encoded.iloc[[top_idx]]
customer_prob = probs[top_idx]

print(f"\nHighest-risk customer (index {top_idx}):")
print(f"  Churn probability: {customer_prob:.2%}")
print(f"\n  Their feature values (non-zero only):")
for col in model.feature_names_in_:
    val = customer[col].values[0]
    if val != 0:
        print(f"    {col:<35} = {val}")

# Individual SHAP explanation
c_shap        = explainer.shap_values(customer)
customer_shap = pd.Series(
    c_shap[1][0],
    index=model.feature_names_in_
).sort_values(key=abs, ascending=False)

print(f"\n  Feature contributions for this customer:")
print(f"  {'Feature':<35} {'SHAP':>8}  {'Effect'}")
print(f"  {'-'*70}")
for feat, val in customer_shap.items():
    if abs(val) < 0.005:
        continue
    effect = "INCREASES churn risk" if val > 0 else "REDUCES  churn risk"
    bar    = ("+" if val > 0 else "-") * min(int(abs(val) * 50), 20)
    print(f"  {feat:<35} {val:>+.4f}   {effect}  {bar}")

# Retention recommendation for this customer
top_risk_factor = customer_shap[customer_shap > 0]
if len(top_risk_factor) > 0:
    biggest_risk = top_risk_factor.index[0]
    print(f"\n  Retention priority: address '{biggest_risk}' first")
    d = directional_results.get(biggest_risk, {})
    print(f"  Recommended action: {d.get('recommended_action', 'investigate')}")

# ── LEVEL 5: SEGMENT DIRECTIONAL SHAP ────────────────────────
print("\n" + "=" * 60)
print("LEVEL 5 — SEGMENT DIRECTIONAL SHAP")
print("Does SHAP direction differ between customer segments?")
print("=" * 60)

# Find categorical columns in the original test data
cat_cols = [c for c in test_df.columns
            if c != TARGET and test_df[c].dtype == object]

# Align test_df index with X_encoded for masking
test_df_reset = test_df.reset_index(drop=True)

for cat_col in cat_cols[:2]:  # limit to first 2 categorical cols
    print(f"\n  Segments by {cat_col}:")
    print(f"  {'Group':<20} {'Top risk feature':<35} {'SHAP':>8}  {'Direction'}")
    print(f"  {'-'*80}")

    for group in sorted(test_df_reset[cat_col].unique()):
        mask = (test_df_reset[cat_col] == group).values

        # Get the sample rows that belong to this group
        sample_reset = sample.reset_index(drop=True)
        # Match sample rows back to test_df_reset
        sample_indices = sample.index.tolist()
        group_in_sample = [
            i for i, idx in enumerate(sample_indices)
            if idx < len(test_df_reset) and test_df_reset[cat_col].iloc[idx] == group
        ]

        if len(group_in_sample) < 5:
            continue

        group_shap = shap_churn[group_in_sample, :]
        group_importance = pd.Series(
            abs(group_shap).mean(axis=0),
            index=model.feature_names_in_
        ).sort_values(ascending=False)

        top_feat       = group_importance.index[0]
        top_importance = group_importance.iloc[0]
        top_col_idx    = list(model.feature_names_in_).index(top_feat)
        top_direction  = group_shap[:, top_col_idx].mean()
        direction_str  = "→ more churn" if top_direction > 0 else "→ less churn"

        print(f"  {str(group):<20} {top_feat:<35} {top_importance:>+.4f}  {direction_str}")

# ── LEVEL 6: BUSINESS ACTION PLAN ────────────────────────────
print("\n" + "=" * 60)
print("LEVEL 6 — BUSINESS ACTION PLAN")
print("What the retention team does with this information")
print("=" * 60)

top5 = list(mean_shap.head(5).items())
business_insights = []

print()
for i, (feature, importance) in enumerate(top5):
    d      = directional_results[feature]
    action = d["recommended_action"]
    direct = d["direction"]

    if direct == "increases_churn":
        when   = "When HIGH — intervene immediately"
        signal = "RISK SIGNAL"
    elif direct == "decreases_churn":
        when   = "When LOW — proactive outreach needed"
        signal = "PROTECTIVE FACTOR"
    else:
        when   = "Monitor per segment"
        signal = "MIXED"

    print(f"  #{i+1} [{signal}] {feature}")
    print(f"      Importance : {importance:.4f}")
    print(f"      Direction  : {d['direction']}")
    print(f"      When to act: {when}")
    print(f"      Action     : {action}")
    print()

    business_insights.append({
        "rank":       i + 1,
        "feature":    feature,
        "importance": round(importance, 4),
        "signal":     signal,
        "direction":  d["direction"],
        "action":     action,
    })

# ── SAVE ALL RESULTS ──────────────────────────────────────────
results = {
    "model":               "churn-model-fayza v2",
    "sample_size":         len(sample),
    "global_importance":   mean_shap.round(4).to_dict(),
    "directional_shap":    directional_results,
    "business_insights":   business_insights,
    "risk_factors":        [f for f, d in directional_results.items()
                            if d["direction"] == "increases_churn"],
    "protective_factors":  [f for f, d in directional_results.items()
                            if d["direction"] == "decreases_churn"],
}

with open("./rai_output/shap/shap_report.json", "w") as f:
    json.dump(results, f, indent=2)

mean_shap.to_csv("./rai_output/shap/feature_importance.csv")

print("=" * 60)
print("Files saved:")
print("  rai_output/shap/global_importance.png")
print("  rai_output/shap/shap_report.json")
print("  rai_output/shap/feature_importance.csv")
print()
print("Wednesday Week 5 COMPLETE")
print()
print(f"KEY FINDING:")
print(f"  Top driver   : {mean_shap.index[0]} (importance={mean_shap.iloc[0]:.4f})")
d0 = directional_results[mean_shap.index[0]]
print(f"  Direction    : {d0['direction']}")
print(f"  Action       : {d0['recommended_action']}")