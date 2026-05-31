"""
Week 11 Thursday — Deploy RAG Flow as Managed Online Endpoint
Same concept as Week 6 churn model deployment.

Week 6 flow:  Model + score.py + env + compute → churn endpoint
Week 11 flow: score.py + env + compute          → RAG endpoint

The difference: no registered model needed because the RAG
pipeline calls APIs directly rather than loading a .pkl file.
"""
import time
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
    Environment,
)
from azure.identity import DefaultAzureCredential

SUBSCRIPTION_ID = "29f1cd2f-d0e2-413e-b913-1976b6924fa6"
RESOURCE_GROUP  = "ai-intern"
WORKSPACE_NAME  = "mal-maverick1"
ENDPOINT_NAME   = "rag-endpoint-fayza"
DEPLOYMENT_NAME = "blue"

ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME
)

# ── Step 1: Create the endpoint ───────────────────────────────
print("── Step 1: Creating endpoint ──")
print(f"  Name: {ENDPOINT_NAME}")
print("  This reserves a permanent HTTPS URL in Azure ML")
print("  Same concept as Week 6 — the address exists before the model")

endpoint = ManagedOnlineEndpoint(
    name=ENDPOINT_NAME,
    description="ConnectPlus RAG pipeline — Week 11 deployment",
    auth_mode="key"
)

try:
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"  Endpoint created: {ENDPOINT_NAME}")
except Exception as e:
    print(f"  Endpoint may already exist: {e}")

# ── Step 2: Get the scoring URI ───────────────────────────────
ep = ml_client.online_endpoints.get(ENDPOINT_NAME)
print(f"\n  Scoring URI: {ep.scoring_uri}")
print(f"  Auth mode:   {ep.auth_mode}")

# ── Step 3: Create deployment environment ────────────────────
print("\n── Step 2: Creating deployment environment ──")
print("  Pinning exact library versions — same as Week 6")

env = Environment(
    name="rag-env-fayza",
    description="RAG pipeline environment",
    conda_file={
        "name": "rag-env",
        "channels": ["defaults", "conda-forge"],
        "dependencies": [
            "python=3.10",
            "pip",
            {"pip": [
                "openai==1.30.1",
                "azure-search-documents==11.4.0",
                "azure-core==1.30.0",
                "azureml-defaults",
            ]}
        ]
    },
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
)

# ── Step 4: Create the deployment ────────────────────────────
print("\n── Step 3: Creating blue deployment ──")
print("  Combining: score.py + environment + compute")
print("  Same blue/green pattern from Week 6")

deployment = ManagedOnlineDeployment(
    name=DEPLOYMENT_NAME,
    endpoint_name=ENDPOINT_NAME,
    code_configuration=CodeConfiguration(
        code="src/week11/thursday_deploy",
        scoring_script="score.py"
    ),
    environment=env,
    instance_type="Standard_DS2_v2",
    instance_count=1
)

print("\n  Deploying... (this takes 5-10 minutes)")
print("  Azure is building a Docker container with your score.py")
print("  Same process as Week 6 — Docker + environment + scoring script")

try:
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    print(f"  Deployment '{DEPLOYMENT_NAME}' succeeded")

    # Route 100% of traffic to blue
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print("  Traffic: 100% → blue")

except Exception as e:
    print(f"  Deployment error: {e}")
    print("  Check Azure ML Studio → Endpoints for details")

# ── Step 5: Get the key ───────────────────────────────────────
print("\n── Step 4: Getting API key ──")
try:
    keys = ml_client.online_endpoints.get_keys(ENDPOINT_NAME)
    primary_key = keys.primary_key
    print(f"  Primary key retrieved (first 8 chars): {primary_key[:8]}...")

    ep = ml_client.online_endpoints.get(ENDPOINT_NAME)
    print(f"\n  LIVE ENDPOINT:")
    print(f"  URL: {ep.scoring_uri}")
    print(f"  Key: {primary_key[:8]}...")
    print("\n  Save these — you need them to call the endpoint")

except Exception as e:
    print(f"  Could not get key: {e}")

print("\n── Deployment complete ──")
