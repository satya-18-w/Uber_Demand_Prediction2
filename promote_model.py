import mlflow
import dagshub
import json
from mlflow import MlflowClient



dagshub.init(repo_owner='satyajitsamal198076', repo_name='Uber_Demand_Prediction2', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/satyajitsamal198076/Uber_Demand_Prediction2.mlflow")


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


# get model name
registered_model_name = 'RandomForestRegressor'
stage = "Staging"

# get the latest version from staging stage
client = MlflowClient()

# get the latest version of model in staging
latest_versions = client.get_latest_versions(name=registered_model_name,stages=[stage])
latest_model_version_staging = latest_versions[0].version

# promotion stage
promotion_stage = "Production"

model_version_prod = client.transition_model_version_stage(
                                                        name=registered_model_name,
                                                        version=latest_model_version_staging,
                                                        stage=promotion_stage,
                                                        archive_existing_versions=True
                                                    )

production_version = model_version_prod.version
new_stage = model_version_prod.current_stage

print(f"The model is moved to the {new_stage} stage having version number {production_version}")