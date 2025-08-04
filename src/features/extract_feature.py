import numpy as np
import pandas as pd
import dask.dataframe as dd
import logging
import yaml
from sklearn.preprocessing import StandardScaler
import pathlib



logger = logging.getLogger("extract_features")
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
    data_path = home_dir.as_posix()+"/data/processed/resampled_data.csv"
    df=pd.read_csv(data_path,parse_dates=["tpep_pickup_datetime"])
    logger.info("Data Read Sucessfully")
    # Extracting the Day and Month columns
    df["day_of_week"]=df["tpep_pickup_datetime"].dt.day_of_week
    df["month"]=df["tpep_pickup_datetime"].dt.month
    df.set_index("tpep_pickup_datetime",inplace=True)
    reg_grp=df.groupby("region")
    pds=[1,2,3,4]
    features=reg_grp["total_pickups"].shift(pds)
    dfs=pd.concat([df,features],axis=1)
    dfs.dropna(how="any",inplace=True)
    
    
    
    # Breaking the data into train(jan,feb) and test(mar)
    train=dfs[dfs["month"].isin([1,2])].drop(columns=["month"])
    test=dfs[dfs["month"].isin([3])].drop(columns=["month"])
    
    
    
    train_data_path=home_dir.as_posix()+"/data/processed/train_data.csv"
    test_data_path=home_dir.as_posix()+"/data/processed/test_data.csv"
    
    train.to_csv(train_data_path,index=True)
    logger.info("Train Data Saved Sucessfully")
    test.to_csv(test_data_path)
    logger.info("Test Data Saved Sucessfully")
    
if __name__ == "__main__":
    main()



