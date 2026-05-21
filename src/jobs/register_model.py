from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import joblib
import os

# ─── 1. CONNECT ───────────────────────────────────────────────────
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# ─── 2. UPDATE TAGS ON YOUR REGISTERED MODEL ──────────────────────
# This is Monday Task 1: tag with stage and accuracy
print("=" * 50)
print("TASK 1: Updating model tags")
print("=" * 50)

model = ml_client.models.get("churn-model-fayza", version="1")

model.tags["stage"]        = "staging"
model.tags["accuracy"]     = "0.95"   
model.tags["f1_score"]     = "0.95"   
model.tags["algorithm"]    = "RandomForest"
model.tags["n_estimators"] = "200"
model.tags["framework"]    = "sklearn"
model.tags["pipeline_run"] = "quiet_forest_g4dltpkr3w"
model.tags["dataset"]      = "churn-dataset@latest"
model.tags["owner"]        = "fayza"
model.tags["week"]         = "5"

updated = ml_client.models.create_or_update(model)
print("Tags updated successfully")
for k, v in updated.tags.items():
    print(f"  {k}: {v}")

# ─── 3. LOAD SPECIFIC VERSION BACK ────────────────────────────────
# This is Monday Task 2: load a specific version into a notebook
print()
print("=" * 50)
print("TASK 2: Loading specific version from registry")
print("=" * 50)

# Get the model metadata from registry by exact version number
model_info = ml_client.models.get("churn-model-fayza", version="1")
print("Retrieved from registry:")
print("  Name:    ", model_info.name)
print("  Version: ", model_info.version)
print("  Stage:   ", model_info.tags.get("stage"))
print("  F1:      ", model_info.tags.get("f1_score"))
print("  Path:    ", model_info.path)

# Download it from Azure Blob Storage to local disk
print()
print("Downloading model to local disk...")
ml_client.models.download(
    name="churn-model-fayza",
    version="1",
    download_path="./monday_model/"
)

# Find the pkl file and load it
print("Loading model...")
for root, dirs, files in os.walk("./monday_model/"):
    for f in files:
        if f.endswith(".pkl"):
            full_path = os.path.join(root, f)
            model = joblib.load(full_path)
            print()
            print("Model loaded successfully")
            print("  Type:          ", type(model).__name__)
            print("  Trees:         ", model.n_estimators)
            print("  Feature count: ", len(model.feature_names_in_))
            print("  Features:      ", model.feature_names_in_.tolist())
            print("  Classes:       ", model.classes_.tolist())

print()
print("Monday Week 5 complete")