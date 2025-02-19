import os
import sys
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"Data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self,excelpath):
        logging.info("Enter the data ingestion method or component")
        try:
            df = pd.read_excel(excelpath)
            logging.info("Read the dataset inside a dataframe is successful")
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Writing the dataframe inside a CSV file from cleansed excel file is successful")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)
            
            logging.info("Train/Test split initiated")
            
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("Train/Test split is completed")
            
            time.sleep(2)  # Wait 1 second before reading
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)



    
    
     
    
