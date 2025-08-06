import streamlit as st
import mlflow
from mlflow.client import MlflowClient
import numpy as np
import pandas as pd
import joblib
import dagshub
import logging
import json
import pathlib
import datetime as dt
from sklearn.pipeline import Pipeline

dagshub.init(repo_owner='satyajitsamal198076', repo_name='Uber_Demand_Prediction2', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/satyajitsamal198076/Uber_Demand_Prediction2.mlflow")





def load_model(run_info):
    model_name=run_info["model_name"]
    model_version=run_info["version"]
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model
    
run_info_path="reports/experiment_info.json"
with open(run_info_path, "r") as f:
            run_info = json.load(f)
     
model=load_model(run_info)

home_dir=pathlib.Path(__file__).parent

kmeans_path=home_dir / "models/mb_kmeans.joblib"
encoder_path=home_dir / "models/encoder.joblib"
scaler_path=home_dir / "models/scaler.joblib"

plot_data_path = home_dir / "data/external/plot_data.csv"
data_path = home_dir / "data/processed/test_data.csv"


kmeans=joblib.load(kmeans_path)
encoder=joblib.load(encoder_path)
scaler=joblib.load(scaler_path)
df_plot = pd.read_csv(plot_data_path)
df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime")



st.title("Uber Demand in New York City 🚕🌆")

st.sidebar.title("Options")

map_type=st.sidebar.radio(label='Select The type of map',options=["Complete Nyc Map","Only the Neighbour Regions"],index=1)
st.subheader("Date")

date=st.date_input("Select the date",value=None,min_value=dt.date(year=2016,month=3,day=1),max_value=dt.date(year=2016,month=3,day=31))

st.write("**Date**",date)


st.subheader("Time")
time = st.time_input("Select the time", value=None)
st.write("**Current Time:**", time)

if date and time:
    d=dt.timedelta(minutes=15)
    next_interval=dt.datetime(year=date.year,month=date.month,day=date.day,hour=time.hour,minute=time.minute)+d
    st.write("**Demand for time :**",next_interval)
    index = pd.Timestamp(f"{date} {next_interval.time()}")
    st.write("**Date & Time:**", index)
    
    st.subheader("Location")
    sample_loc=df_plot.sample(1).reset_index(drop=True)
    lat=sample_loc["pickup_latitude"].item()
    long=sample_loc["pickup_latitude"].item()
    region=sample_loc["region"].item()
    st.write("** Your Current Location **")
    st.write("**Lat: {lat} **")
    st.write("** Long: {long} **")
    
    with st.spinner("Fetching Your Current Region"):
        st.sleep(3)
        
    st.write("** Region ID: {region} **")
    
    
    scaled_cord=scaler.transform(sample_loc.iloc[:,0:2])
    
    st.subheader("MAP")
    
    colors = ["#FF0000", "#FF4500", "#FF8C00", "#FFD700", "#ADFF2F", 
              "#32CD32", "#008000", "#006400", "#00FF00", "#7CFC00", 
              "#00FA9A", "#00FFFF", "#40E0D0", "#4682B4", "#1E90FF", 
              "#0000FF", "#0000CD", "#8A2BE2", "#9932CC", "#BA55D3", 
              "#FF00FF", "#FF1493", "#C71585", "#FF4500", "#FF6347", 
              "#FFA07A", "#FFDAB9", "#FFE4B5", "#F5DEB3", "#EEE8AA"]
    region_colors={ region:colors[i]  for i,region in enumerate(df_plot["region"].unique())}
    df_plot["color"]=df_plot["region"].map(region_colors)
    
    pipeline=Pipeline(
        [
            ("encoder",encoder),
            ("reg",model)
        ]
    )
    
    
    if map_type == "Complete Nyc Map":
        progress_bar=st.progress(value=0,text="Operation in Progress. Please wait... ")
        
        for percent_complete in range(100):
            st.sleep(0.05)
            progress_bar.progress(percent_complete + 1, text="Operation in progress. Please wait.")
            
        
        st.map(data=df_plot,latitude="pickup_latitude",longitude="pickup_longitude",color="color",size=0.01)
        
        progress_bar.empty()
        
        input_data=df.loc[index,:].sort_values("region")
        target=input_data["total_pickups"]
        
        predictions=pipeline.predict(input_data.drop(columns=["total_pickups"]))
        st.markdown("### Map Legend")
        for ind in range(0,30):
                     color = colors[ind]
                     demand = predictions[ind]
                     if region == ind:
                            region_id = f"{ind} (Current region)"
                     else:
                            region_id = ind
                     st.markdown(
                     f'<div style="display: flex; align-items: center;">'
                     f'<div style="background-color:{color}; width: 20px; height: 10px; margin-right: 10px;"></div>'
                     f'Region ID: {region_id} <br>'
                     f"Demand: {int(demand)} <br> <br>", unsafe_allow_html=True
                     )
                     
    elif map_type == "Only the Neighbour Regions":
        distances=kmeans.transform(scaled_cord).values.ravel().tolist()
        
        distances=list(enumerate(distances))
        sorted_distances = sorted(distances, key=lambda x: x[1])[0:9]
        indexes = sorted([ind[0] for ind in sorted_distances])
        
        df_plot_filtered=df_plot[df_plot["region"].isin(indexes)]
        
        progress_bar = st.progress(value=0,text="Operation in progress. Please wait.")
        for percent_complete in range(100):
                     dt.sleep(0.05)
                     progress_bar.progress(percent_complete + 1, text="Operation in progress. Please wait.")
              
              # map
        st.map(data=df_plot_filtered, latitude="pickup_latitude", 
                     longitude="pickup_longitude", size=0.01,
                     color="color")
              
              
        progress_bar.empty()
              
              # filter the data 
        input_data = df.loc[index, :]
        input_data = input_data.loc[input_data["region"].isin(indexes), :].sort_values("region")
        target = input_data["total_pickups"]

              # do the predictions
        predictions = pipeline.predict(input_data.drop(columns=["total_pickups"]))
              
              # show the map labels
              # Display the map legend
        st.markdown("### Map Legend")
        for ind in range(0,9):
                     color = colors[indexes[ind]]
                     demand = predictions[ind]
                     if region == indexes[ind]:
                            region_id = f"{indexes[ind]} (Current region)"
                     else:
                            region_id = indexes[ind]
                     st.markdown(
                     f'<div style="display: flex; align-items: center;">'
                     f'<div style="background-color:{color}; width: 20px; height: 10px; margin-right: 10px;"></div>'
                     f'Region ID: {region_id} <br>'
                     f"Demand: {int(demand)} <br> <br>", unsafe_allow_html=True
              )
       
       