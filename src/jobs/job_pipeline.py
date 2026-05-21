from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

COMPUTE = "cpu-cluster-fayza"
ENV     = "AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1"

data_prep_component = command(
    name="data_prep",
    display_name="Data Preparation",
    code="./src/pipelines",
    command=(
        "python data_prep.py "
        "--input_data ${{inputs.raw_data}} "
        "--output_data ${{outputs.cleaned_data}}/cleaned.csv"
    ),
    inputs={"raw_data": Input(type=AssetTypes.URI_FILE)},
    outputs={"cleaned_data": Output(type=AssetTypes.URI_FOLDER)},
    environment=ENV,
    is_deterministic=False,
)

train_component = command(
    name="train",
    display_name="Train Model",
    code="./src/pipelines",
    command=(
        "python train.py "
        "--input_data ${{inputs.train_data}}/cleaned.csv "
        "--model_output ${{outputs.model}}/model.pkl"
    ),
    inputs={"train_data": Input(type=AssetTypes.URI_FOLDER)},
    outputs={"model": Output(type=AssetTypes.URI_FOLDER)},
    environment=ENV,
    is_deterministic=False,
)

evaluate_component = command(
    name="evaluate",
    display_name="Evaluate Model",
    code="./src/pipelines",
    command=(
        "python evaluate.py "
        "--model_path ${{inputs.model}}/model.pkl "
        "--data_path ${{inputs.data}}/cleaned.csv"
    ),
    inputs={
        "model": Input(type=AssetTypes.URI_FOLDER),
        "data":  Input(type=AssetTypes.URI_FOLDER),
    },
    environment=ENV,
    is_deterministic=False,
)

@pipeline(
    display_name="Churn Prediction Pipeline",
    description="prep -> train -> evaluate",
    default_compute=COMPUTE,
)
def churn_pipeline(raw_data: Input(type=AssetTypes.URI_FILE)):
    prep_step = data_prep_component(raw_data=raw_data)
    train_step = train_component(train_data=prep_step.outputs.cleaned_data)
    _ = evaluate_component(
        model=train_step.outputs.model,
        data=prep_step.outputs.cleaned_data,
    )
    return {"trained_model": train_step.outputs.model}

pipeline_job = churn_pipeline(
    raw_data=Input(
        type=AssetTypes.URI_FILE,
        path="azureml://subscriptions/29f1cd2f-d0e2-413e-b913-1976b6924fa6/resourcegroups/ai-intern/workspaces/AIintern/datastores/workspaceblobstore/paths/customer_churn_dataset-testing-master.csv"
    )
)

submitted = ml_client.jobs.create_or_update(
    pipeline_job,
    experiment_name="churn-pipeline-run"
)

print("Submitted:", submitted.name)
print("Studio URL:", submitted.studio_url)
ml_client.jobs.stream(submitted.name)