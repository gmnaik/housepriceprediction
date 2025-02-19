import os
import sys
import pandas as pd
import papermill as pm

#from pathlib import Path
#sys.path.append(str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataNotebookConfig:
    input_notebook_path: str=os.path.join('notebook',"House_Price_Prediction_Imputation_EDA.ipynb")
    output_notebook_path: str=os.path.join('notebook',"House_Price_Prediction_Imputation_EDA.ipynb")
    
class DataNotebook:
    def __init__(self):
        self.NotebookIngestionConfig = DataNotebookConfig()
    
    def run_notebook(self):
        try:
            # Execute the notebook
            pm.execute_notebook(
                self.NotebookIngestionConfig.input_notebook_path,           # Input notebook file path
                self.NotebookIngestionConfig.output_notebook_path            # Output notebook file path (for debugging)
            )
            
            cleaned_excel_path = os.path.join('artifacts',"Property_Cleansed.xlsx")
            return cleaned_excel_path
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj=DataNotebook()
    excelpath = obj.run_notebook()
    print("Cleaned data file successfully generated at path:",excelpath)
    
    data_ingestion = DataIngestion()
    train_path,test_path = data_ingestion.initiate_data_ingestion(excelpath)
    print("Data Ingestion is successful")
    
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_path,test_path)
    
    print("Data transformation is successful")
    
    model_trainer = ModelTrainer()
    bestmodel,bestmodelaccuracy = model_trainer.initiate_model_trainer(train_arr,test_arr)
    
    print("bestmodel:",bestmodel)
    print("bestmodelaccuracy:",bestmodelaccuracy)
    
    print("Data modeling is successful")
    