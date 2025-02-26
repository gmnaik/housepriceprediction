import os
import sys
import pandas as pd
import papermill as pm

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataNotebookConfig:
    input_notebook_path: str=os.path.join('notebook',"House_Price_Prediction_Imputation_EDA.ipynb")
    output_notebook_path: str=os.path.join('notebook',"House_Price_Prediction_Imputation_EDA.ipynb")
    
class DataNotebook:
    def __init__(self):
        self.NotebookIngestionConfig = DataNotebookConfig()
    
    def run_notebook(self):
        logging.info("Execute jupyter notebook file to impute missing values and add cleaned dataset in Property_Cleansed.xlsx file")
        try:
            # Execute the notebook
            pm.execute_notebook(
                self.NotebookIngestionConfig.input_notebook_path,           # Input notebook file path
                self.NotebookIngestionConfig.output_notebook_path            # Output notebook file path (for debugging)
            )
            
            cleaned_excel_path = os.path.join('artifacts',"Property_Cleansed.xlsx")
            
            logging.info("Execution of jupyter notebook is successful and cleaned dataset is added in Property_Cleansed.xlsx file")
            
            return cleaned_excel_path
        
        except Exception as e:
            raise CustomException(e,sys)
        

    