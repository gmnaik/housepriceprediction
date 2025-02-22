import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")
    y_scaled_file_path = os.path.join('artifacts', "scaler_y.pkl")

class MultiColumnLabelEncoder(TransformerMixin):
    def fit(self, X, y=None):
        return self  # No fitting required

    def transform(self, X):
        # Convert X to DataFrame if it's a NumPy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)  # Convert to DataFrame

        X_encoded = X.copy()  # Copy to avoid modifying original data
        for col in X_encoded.columns:
            X_encoded[col], _ = pd.factorize(X_encoded[col])  # Apply pd.factorize()
        return X_encoded
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self,numerical_columns,categorical_columns):
        '''
        This function is responsible for data transformation
        '''
        try:       
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
                )
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("label_encoder", MultiColumnLabelEncoder())
                ]
                )
            
            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            train_df = train_df.drop(columns=['Unnamed: 0'], errors='ignore')
            
            numerical_columns = [feature for feature in train_df.columns if train_df[feature].dtype != 'O']               # 0 means object
            categorical_columns = [feature for feature in train_df.columns if train_df[feature].dtype == 'O']
            numerical_columns.remove("Price_Millions")
            
            logging.info("Split dependent and independent variables")
            X_train = train_df.drop('Price_Millions',axis = 1)
            y_train = train_df['Price_Millions']
            
            X_test = test_df.drop('Price_Millions',axis = 1)
            y_test = test_df['Price_Millions']
            
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object(numerical_columns,categorical_columns)
            
            logging.info("Applying preprocessing object on training dataframe and testing dataframe")
            
            X_train = preprocessing_obj.fit_transform(X_train)
            X_test = preprocessing_obj.transform(X_test)
            
            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
            y_test = scaler_y.transform(y_test.values.reshape(-1, 1)) 

            train_arr = np.c_[X_train,np.array(y_train)]
            test_arr = np.c_[X_test,np.array(y_test)]
            
            logging.info("Saved preprocessing object.")
            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj)
            
            logging.info("Saved y-scaler object.")
            save_object(file_path = self.data_transformation_config.y_scaled_file_path,obj=scaler_y)
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
            