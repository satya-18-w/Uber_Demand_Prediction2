import pytest
import mlflow
import dagshub
import json
from pathlib import Path
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn import set_config
import joblib


set_config(transform_output="pandas")



def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


# model_path = load_model_information("reports/experiment_info.json")["model_uri"]


# model = mlflow.sklearn.load_model(model_path)
model=joblib.load("models/best_model.joblib")


current_path = Path(__file__)

root_path = current_path.parent.parent

train_data_path = root_path / "data/processed/train_data.csv"
test_data_path = root_path / "data/processed/test_data.csv"


encoder_path = root_path / "models/encoder.joblib"
encoder = joblib.load(encoder_path)


model_pipe = Pipeline(steps=[
    ("encoder",encoder),
    ("regressor",model)
])

# test function
@pytest.mark.parametrize(argnames="data_path,threshold",
                         argvalues=[(train_data_path,0.1),
                                    (test_data_path,0.1)])
def test_performance(data_path, threshold):
    # load the data from path
    data = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime")
    # make X and y
    X = data.drop(columns=["total_pickups"])
    y = data["total_pickups"]
    # do predictions
    y_pred = model_pipe.predict(X)
    # calculate the loss
    loss = mean_absolute_percentage_error(y, y_pred)
    # check the performance
    assert loss <= threshold,  f"The model does not pass the performance threshold of {threshold}"