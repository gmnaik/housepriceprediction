import os
import sys
import pandas as pd
import papermill as pm

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.components.data_notebook import DataNotebook
from src.components.data_notebook import DataNotebookConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

        
if __name__ == "__main__":
    data_notebook=DataNotebook()
    excelpath = data_notebook.run_notebook()
    logging.info("Notebook execution is successful.")
    
    data_ingestion = DataIngestion()
    train_path,test_path = data_ingestion.initiate_data_ingestion(excelpath)
    logging.info("Data Ingestion is successful.")
    
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_path,test_path)
    logging.info("Data transformation is successful.")
    
    model_trainer = ModelTrainer()
    bestmodel,bestmodelaccuracy = model_trainer.initiate_model_trainer(train_arr,test_arr)
    logging.info("Data modeling is successful.")
    
    print("bestmodel:",bestmodel)
    print("bestmodelaccuracy:",bestmodelaccuracy)
    