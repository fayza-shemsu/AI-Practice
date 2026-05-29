from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment

# =========================
# 1. CONNECT TO AZURE
# =========================
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="29f1cd2f-d0e2-413e-b913-1976b6924fa6",
    resource_group_name="ai-intern",
    workspace_name="AIintern",
)

# =========================
# 2. ENVIRONMENT
# =========================
env = Environment(
    name="churn-env",
    conda_file="environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
)

# =========================
# 3. JOB (WITH REGISTERED DATA)
# =========================
job = command(
    code=".",

    # IMPORTANT: pass dataset into script
    command="python src/models/train.py --data_path ${{inputs.data}}",

    inputs={
        "data": Input(
            type="uri_file",
            path="azureml:churn-dataset:1"   
        )
    },

    environment=env,
    compute="cpu-cluster",
    display_name="churn-training-job"
)

# =========================
# 4. SUBMIT JOB
# =========================
returned_job = ml_client.jobs.create_or_update(job)

print(f"Job submitted: {returned_job.name}")