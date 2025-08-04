import joblib
from sklearn.metrics import mean_absolute_percentage_error
import yaml
import mlflow
import numpy as np
import pandas as pd
import logging
import pathlib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import set_config
import optuna
import dagshub
import sys
import json

import dagshub
dagshub.init(repo_owner='satyajitsamal198076', repo_name='Uber_Demand_Prediction2', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/satyajitsamal198076/Uber_Demand_Prediction2.mlflow")
set_config(transform_output="pandas")


logger = logging.getLogger("Model Training")
logger.setLevel(logging.INFO)

# attach a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# make a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

def main():
    curr_dir=pathlib.Path(__file__)
    home_dir=curr_dir.parent.parent.parent
    mlflow.set_experiment("Model_Training")
    train_path=sys.argv[1]
    test_path=sys.argv[2]
    encoder=ColumnTransformer(
        [
            ("one_hot_encoder",OneHotEncoder(sparse_output=False,drop="first"),["day_of_week","region"])
            
        ],
        remainder="passthrough"
    )
    train=pd.read_csv(train_path)
    train.set_index("tpep_pickup_datetime",inplace=True)
    X_train=train.drop(columns=["total_pickups"])
    y_train=train["total_pickups"]
    
    
    test=pd.read_csv(test_path)
    test.set_index("tpep_pickup_datetime",inplace=True)
    X_test=test.drop(columns=["total_pickups"])
    y_test=test["total_pickups"]
    
    
    X_train_encoded=encoder.fit_transform(X_train)
    X_test_encoded=encoder.fit_transform(X_test)
    
    
    model_path=home_dir.as_posix() + "/models/best_model.joblib"
    encoder_path= home_dir.as_posix() + "/models/encoder.joblib"
    joblib.dump(encoder,encoder_path)
    
    def objective(trial):
        
        with mlflow.start_run(nested=True) as child:
            
            n_estimators_rf=trial.suggest_int("n_estimators_rf",10,100,step=10)
            max_depth_rf=trial.suggest_int("max_depth_rf",3,15)
            model=RandomForestRegressor(n_estimators=n_estimators_rf,max_depth=max_depth_rf,n_jobs=-1,random_state=42)
            mlflow.log_params(model.get_params())
            
            model.fit(X_train_encoded,y_train)
            y_pred=model.predict(X_test_encoded)
            mape=mean_absolute_percentage_error(y_test,y_pred)
            mlflow.log_metric("MAPE",mape)
            
            
            return mape
    with mlflow.start_run(run_name="BEST_MODEL",nested=True) as parent:
        study=optuna.create_study(study_name="Model_selection",direction="minimize")
        study.optimize(objective,n_trials=50,n_jobs=-1)
        mlflow.set_tag("author","Satyajit")
        mlflow.set_tag("Model","Random Forest")
        
        mlflow.log_params(study.best_params)
        
        mlflow.log_metric("BEST_MAPE",study.best_value)
        best_model=RandomForestRegressor(**study.best_params)
        best_model.fit(X_train_encoded,y_train)
        y_pred=best_model.predict(X_test_encoded)
        mape=mean_absolute_percentage_error(y_test,y_pred)
        mlflow.log_metric("MAPE",mape)
        signature=mlflow.modells.infer_signature(X_train_encoded,y_train)
        mlflow.sklearn.log_model(best_model,"Best_Model",signature=signature,input_example=X_train_encoded)
        train_df=mlflow.data.from_pandas(train)
        test_df=mlflow.data.from_pandas(test)
        mlflow.log_input(train_df,"Training Data")
        mlflow.log_input(test_df,"Testing Data")
        
        
        # Storing the best_model
        joblib.dump(best_model,model_path)
        
        run_id=parent.info.run_id
        model_uri=f"runs:/{run_id}/Model"
        model_verision=mlflow.register_model(model_uri,"gredient_boosting_sklearn")
        print(parent.info)
        info={"run_id":run_id,"Model_path":"Model","model_uri":model_uri}
        with open("reports/experiment_info.json","w") as f:
            
            
            
            json.dump(info,f,indent=4)
            
            
            
            
if __name__ == "__main__":
    main()
            
    
    
    
    
    
    
    