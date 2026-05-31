# =========================
# 1. IMPORTS
# =========================
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, automl, Input, Output

# =========================
# 2. CONNECT TO AZURE
# =========================
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="29f1cd2f-d0e2-413e-b913-1976b6924fa6",
    resource_group_name="ai-intern",
    workspace_name="AIintern",
)

# =========================
# 3. LOAD DATA ASSET
# =========================
data = ml_client.data.get("churn_clean", version="1")

# =========================
# 4. DEFINE AUTOML JOB
# =========================
# 4. DEFINE AUTOML JOB
# =========================
job = automl.classification(
    compute="cpu-cluster-fayza",
    experiment_name="churn_automl_sdk_v1",
    training_data=Input(
        type="mltable",
        path=data.id
    ),
    target_column_name="Churn",
    primary_metric="AUC_weighted",
)

# =========================
# 5. SET LIMITS (VERY IMPORTANT)
# =========================
job.set_limits(
    max_trials=30,                # number of models to try
    max_concurrent_trials=1,      # parallel runs
    timeout_minutes=60            # stop after 60 mins
)

# =========================
# 7. SUBMIT JOB
# =========================
returned_job = ml_client.jobs.create_or_update(job)

print(f"\n🚀 Job submitted: {returned_job.name}")

# =========================
# 8. STREAM LOGS (LIVE MONITORING)
# =========================
ml_client.jobs.stream(returned_job.name)

# =========================
# 9. GET BEST RUN
# =========================
parent_job = ml_client.jobs.get(returned_job.name)

best_child_run_id = parent_job.properties["best_child_run_id"]

print("\n🏆 Best child run:", best_child_run_id)

# =========================
# 10. DOWNLOAD ALL OUTPUTS
# =========================
ml_client.jobs.download(
    name=returned_job.name,
    download_path="./outputs"
)

print("\n📁 Outputs downloaded to ./outputs")