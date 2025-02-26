import os
import sys
import pytest
import numpy as np

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException

# Fixture for DataTransformation
@pytest.fixture
def data_transformation():
    return DataTransformation()

# Fixture for ModelTrainer
@pytest.fixture
def model_trainer():
    return ModelTrainer()

# Test successful model training
def test_successful_model_training(model_trainer, data_transformation):
    try:
        # Provide valid train and test CSV file paths
        train_path = os.path.join("artifacts","train.csv")
        test_path = os.path.join("artifacts","test.csv")
        
        train_array,test_array, _ = data_transformation.initiate_data_transformation(train_path, test_path)
        
        # Call the method
        best_model_name, best_model_accuracy = model_trainer.initiate_model_trainer(train_array, test_array)
        
        # Check if the best model is selected and saved
        assert best_model_name is not None
        assert best_model_accuracy > 0.6
        
        assert os.path.exists("artifacts/model.pkl")
        
    except Exception as e:
        raise CustomException(e, sys)


# Test handling of invalid input data
def test_invalid_input_data(model_trainer):
    try:
        # Provide empty or invalid train and test arrays
        train_array = np.array([])
        test_array = np.array([])
        
        # Call the method and expect an exception
        with pytest.raises(Exception):
            model_trainer.initiate_model_trainer(train_array, test_array)
            
    except Exception as e:
        raise CustomException(e, sys)

# Test handling of no best model found
def test_no_best_model_found(model_trainer):
    try:
        # Provide train and test arrays that result in poor model performance
        train_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        test_array = np.array([[10, 11, 12], [13, 14, 15]])
        
        # Call the method and expect an exception
        with pytest.raises(CustomException):
            model_trainer.initiate_model_trainer(train_array, test_array)
            
    except Exception as e:
        raise CustomException(e, sys)