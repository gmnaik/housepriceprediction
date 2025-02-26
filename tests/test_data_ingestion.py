import os
import sys
import pytest
import pandas as pd

from src.components.data_notebook import DataNotebook
from src.components.data_ingestion import DataIngestion
from src.exception import CustomException

# Fixture for DataIngestion
@pytest.fixture
def data_notebook():
    return DataNotebook()

@pytest.fixture
def data_ingestion():
    return DataIngestion()

# Test successful data ingestion
def test_successful_data_ingestion(data_notebook, data_ingestion):
    try:
        # Provide a valid Excel file path
        cleaned_excel_path = data_notebook.run_notebook()
        #excel_path = "test_data.xlsx"
        
        # Call the method
        train_path, test_path = data_ingestion.initiate_data_ingestion(cleaned_excel_path)
        
        # Check if the files are created
        assert os.path.exists(train_path)
        assert os.path.exists(test_path)
        
        # Check if the files are not empty
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        assert not train_df.empty
        assert not test_df.empty
        
    except Exception as e:
        raise CustomException(e, sys)

