import os
import sys
import pytest
import numpy as np
import pandas as pd
from src.components.data_transformation import DataTransformation
from src.exception import CustomException

# Fixture for DataTransformation
@pytest.fixture
def data_transformation():
    return DataTransformation()

# Test successful data transformation
def test_successful_data_transformation(data_transformation):
    try:
        # Provide valid train and test CSV file paths
        train_path = os.path.join("artifacts","train.csv")
        test_path = os.path.join("artifacts","test.csv")
        
        # Call the method
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)
        
        # Check if the transformed data arrays are not empty
        assert train_arr.size > 0
        assert test_arr.size > 0
        
    except Exception as e:
        raise CustomException(e, sys)

# Test handling of missing values
def test_missing_values_handling(data_transformation):
    try:
        # Provide train and test CSV files with missing values
        train_path = os.path.join("artifacts","train.csv")
        test_path = os.path.join("artifacts","test.csv")
        
        # Call the method
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)
        
        # Check if the transformed data arrays do not contain NaN values
        assert not np.isnan(train_arr).any()
        assert not np.isnan(test_arr).any()
        
    except Exception as e:
        raise CustomException(e, sys)
    
    
#Test whether preprocessing and y_scaled pickle file is created
def test_pickle_file_creation(data_transformation):
    try:
        # Provide train and test CSV files with missing values
        train_path = os.path.join("artifacts","train.csv")
        test_path = os.path.join("artifacts","test.csv")
        
        # Call the method
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)
        
        # Check if preprocessed object and y_scaled values is saved
        assert os.path.exists("artifacts/preprocessor.pkl")
        assert os.path.exists("artifacts/scaler_y.pkl")
        
    except Exception as e:
        raise CustomException(e, sys)
    

