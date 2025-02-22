import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        print("features:\n",features)
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            yscaled_path = 'artifacts\scaler_y.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            y_inversescaled = load_object(file_path = yscaled_path)
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            print("preds inside predict function",preds)
            
            actualpredictedprice = y_inversescaled.inverse_transform(np.array(preds[0]).reshape(-1, 1))
            print("Actual predicted price",actualpredictedprice[0][0])
            return round(actualpredictedprice[0][0],2)
        
        except Exception as e:
            raise CustomException(e,sys)
    

class CustomData:
    def __init__(self, Floor_No:int,Units_Available:int,Covered_Area:int,Carpet_Area:int,
                 Sqft_Price:int,Total_Amenities:int,Area_Difference:int,Floors:int,
                 PossessionStatus:str,FlooringType:str,Society:str,
                 FurnishedType:str,Facing:str,Transaction_Type:str,Type_of_Property:str,
                 City:str,Bathroom:str,Parking:str,Bedroom:str,Balconies:str,Ownership_Type:str):
        self.Floor_No = Floor_No
        self.Units_Available = Units_Available
        self.Covered_Area = Covered_Area
        self.Carpet_Area = Carpet_Area
        self.Sqft_Price = Sqft_Price
        self.Total_Amenities = Total_Amenities
        self.Area_Difference = Area_Difference
        self.Floors = Floors
        self.PossessionStatus = PossessionStatus
        self.FlooringType = FlooringType
        self.Society = Society
        self.FurnishedType = FurnishedType
        self.Facing = Facing
        self.Transaction_Type = Transaction_Type
        self.Type_of_Property = Type_of_Property
        self.City = City
        self.Bathroom = Bathroom
        self.Parking = Parking
        self.Bedroom = Bedroom
        self.Balconies = Balconies
        self.Ownership_Type = Ownership_Type
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Floor No" : [self.Floor_No],
                "Units Available" : [self.Units_Available],
                "Covered Area" : [self.Covered_Area],
                "Carpet Area" : [self.Carpet_Area],
                "Sqft Price" : [self.Sqft_Price],
                "Total Amenities" : [self.Total_Amenities],
                "Area Difference (%)" : [self.Area_Difference],
                "Floors" : [self.Floors],
                "Possession Status" : [self.PossessionStatus],
                "Flooring Type" : [self.FlooringType],
                "Society" : [self.Society],
                "Furnished Type" : [self.FurnishedType],
                "Facing" : [self.Facing],
                "Transaction Type" : [self.Transaction_Type],
                "Type of Property" : [self.Type_of_Property],
                "City" : [self.City],
                "Bathroom" : [self.Bathroom],
                "Parking" : [self.Parking],
                "Bedroom" : [self.Bedroom],
                "Balconies" : [self.Balconies],
                "Ownership Type" : [self.Ownership_Type]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)