import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,load_object

from sklearn.metrics import roc_curve, roc_auc_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def evaluate_model(self,true, predicted):
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mean_squared_error(true, predicted))
        r2_square = r2_score(true, predicted)
        return mae, rmse, r2_square
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            
            y_train = y_train.astype(float)
            y_test = y_test.astype(float)
            
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(random_state=42,enable_categorical=True), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            param_grids = {
                "Linear Regression": {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                    'solver': ['saga'],
                    'class_weight' : ['balanced'],
                },
                "Lasso": {
                    'solver': ['svd', 'lsqr', 'eigen'],
                    'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]  # Only valid for 'lsqr' or 'eigen'
                },
                "Ridge": {
                    'criterion': ['gini', 'entropy'],
                    'class_weight' : ['balanced'],
                    'max_depth': [3, 5, 10,12],
                    'min_samples_split': [1, 5, 10],
                    'min_samples_leaf': [1, 2, 5],
                    'ccp_alpha': [0.0, 0.01, 0.1]
                },
                "K-Neighbors Regressor": {
                    'n_estimators': [20,30,50, 100],
                    'max_depth': [3, 5,11],
                    'min_samples_split': [1, 5, 11],
                    'min_samples_leaf': [1, 2, 5],
                    'class_weight' : ['balanced', 'balanced_subsample']
                },
                
                "Decision Tree": {
                    'n_estimators': [50, 70],
                    'num_leaves': [31, 50],
                    'max_depth': [10, 20],
                    'learning_rate': [0.01, 0.1],
                    'is_unbalance': [True, False],  
                    'scale_pos_weight': [1, 10]
                },
                "Random Forest Regressor": {
                    'iterations': [200, 230],
                    'depth': [6, 9],
                    'learning_rate': [0.1,0.2],
                    'class_weights': [[1, 5], [1, 10]] 
                },
                
                "XGBRegressor": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto'],
                    'class_weight': ['balanced']
                },
                "CatBoosting Regressor": {
                    'n_neighbors': [7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
                
            }
            
            model_list = []
            r2score_list =[]
            
            for i in range(len(list(models))):
                model = list(models.values())[i]
                #para = param_grids[list(models.keys())[i]]
                model.fit(X_train, y_train) # Train model
                
                #gs = GridSearchCV(model,para,cv=3,scoring='f1_weighted', n_jobs=-1)
                #gs.fit(X_train,y_train)
                
                #model = gs.best_estimator_
                #model.set_params(**gs.best_params_)
                #model.fit(X_train,y_train)
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Evaluate Train and Test dataset
                model_train_mae , model_train_rmse,model_train_r2score = self.evaluate_model(y_train, y_train_pred)

                model_train_mae , model_test_rmse,model_test_r2score = self.evaluate_model(y_test, y_test_pred)

                print(list(models.keys())[i])
                model_list.append(list(models.keys())[i])
                
                print('Model performance for Training set')
                print("- RMSE score: \n{:.4f}".format(model_train_rmse))
                print("- R2 score report:\n {}".format(model_train_r2score))
                
                
                print('Model performance for Test set')
                print("- RMSE score: \n{:.4f}".format(model_test_rmse))
                print("- R2 score report: \n{}".format(model_test_r2score))
                
                print('-------------------------------------------------------------------------------------------')
               
                r2score_list.append(model_test_r2score)
            
            model_dict = {}

            for i in range(0,len(model_list)):
                model_dict[model_list[i]] = r2score_list[i]
            
            print("model_dict:",model_dict)

            best_model_score = max(sorted(model_dict.values()))
            
            #To get best model name from dictionary
            best_model_name = list(model_dict.keys())[list(model_dict.values()).index(best_model_score)]
                        
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            else:
                pass
            
            logging.info("Best model found")
            
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model) 
            
            return best_model_name,model_dict[best_model_name]
        
        except Exception as e:
            raise CustomException(e,sys)