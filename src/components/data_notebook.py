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
    
    #prop_df = pd.read_excel(excelpath)

    print("Data Ingestion is successful")