from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

ENDPOINT_NAME = "churn-endpoint-fayza"
MODEL_NAME    = "churn-model-fayza"
MODEL_VERSION = "2"

# STEP 1 — CREATE THE ENDPOINT
print("Creating endpoint:", ENDPOINT_NAME)
endpoint = ManagedOnlineEndpoint(
    name=ENDPOINT_NAME,
    description="Churn prediction endpoint — Fayza Week 6",
    auth_mode="key",
    tags={"owner": "fayza", "model": "churn-model-fayza", "week": "6"}
)
endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("Endpoint created:", endpoint.name)
print("State:", endpoint.provisioning_state)

# STEP 2 — CREATE THE DEPLOYMENT
print()
print("Creating deployment: blue (this takes 5-10 minutes)...")
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=ENDPOINT_NAME,
    model=MODEL_NAME + ":" + MODEL_VERSION,
    code_configuration=CodeConfiguration(
        code="src/scoring",
        scoring_script="score.py"
    ),
    environment=Environment(
        conda_file="envs/churn_env.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    ),
    instance_type="Standard_DS2_v2",
    instance_count=1,
)
deployment = ml_client.online_deployments.begin_create_or_update(deployment).result()
print("Deployment created:", deployment.name)

# STEP 3 — ROUTE ALL TRAFFIC TO BLUE
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("Traffic: 100% to blue")

# STEP 4 — GET URL AND KEY
endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
keys     = ml_client.online_endpoints.get_keys(ENDPOINT_NAME)

print()
print("=" * 60)
print("DEPLOYMENT COMPLETE")
print("=" * 60)
print("Endpoint URL:", endpoint.scoring_uri)
print("API Key:     ", keys.primary_key[:20], "...")
print()
print("Save these — you need them to call the endpoint")
print("Tuesday Week 6 COMPLETE")
