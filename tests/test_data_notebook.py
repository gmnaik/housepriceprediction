import os
import sys
import pytest
import pandas as pd

from src.components.data_notebook import DataNotebook
from src.exception import CustomException

# Fixture for data notebook
@pytest.fixture
def data_notebook():
    return DataNotebook()


def test_data_notebook_execution(data_notebook):
    try:
        cleaned_excel_path = data_notebook.run_notebook()
        
        #Check if cleaned excel file is created
        assert os.path.exists(cleaned_excel_path)
        
        # Check if the file is not empty
        assert os.path.getsize(cleaned_excel_path) > 0
        
    except Exception as e:
        raise CustomException(e, sys)
    
    

def test_data_notebook_cleaned_data(data_notebook):
    try:
        # Run the notebook execution
        cleaned_excel_path = data_notebook.run_notebook()
        
        # Load the cleaned Excel file
        df = pd.read_excel(cleaned_excel_path)
        
        # Check if the DataFrame is not empty
        assert not df.empty
        
        # Check if the DataFrame has the expected columns (adjust based on your dataset)
        expected_columns = ['Floor No','Units Available','Covered Area','Carpet Area','Sqft Price','Total Amenities','Area Difference (%)','Floors','Price_Millions','Possession Status','Flooring Type','Society','Furnished Type','Facing','Transaction Type','Type of Property','City','Bathroom','Parking','Bedroom','Balconies','Ownership Type']  # Replace with actual column names
        assert all(column in df.columns for column in expected_columns)
        
    except Exception as e:
        raise CustomException(e, sys)