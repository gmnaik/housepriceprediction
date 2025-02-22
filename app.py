import pickle
from flask import Flask, request, render_template,jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS,cross_origin
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

#Route for  a home page
@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint_dl():
    if request.method == "GET":
        return render_template('index.html')
    else:
        data = CustomData(
            Floor_No=int(request.form.get('Floor No')),
            Units_Available=int(request.form.get('Units Available')),
            Covered_Area=int(request.form.get('Covered Area')),
            Carpet_Area=int(request.form.get('Carpet Area')),
            Sqft_Price=int(request.form.get('Sqft Price')),
            Total_Amenities=int(request.form.get('Total Amenities')),
            Area_Difference=int(request.form.get('Area Difference (%)')),
            Floors=int(request.form.get('Floors')),
            PossessionStatus=request.form.get('Possession Status'),
            FlooringType=request.form.get('Flooring Type'),
            Society=request.form.get('Society'),
            FurnishedType=request.form.get('Furnished Type'),
            Facing=request.form.get('Facing'),
            Transaction_Type=request.form.get('Transaction Type'),
            Type_of_Property=request.form.get('Type of Property'),
            City=request.form.get('City'),
            Bathroom=request.form.get('Bathroom'),
            Parking=request.form.get('Parking'),
            Bedroom=request.form.get('Bedroom'),
            Balconies=request.form.get('Balconies'),
            Ownership_Type=request.form.get('Ownership Type')
        )
        
        pred_df = data.get_data_as_data_frame()
        print("pred_df:\n",pred_df)
        predict_pipeline = PredictPipeline()
        
        results = predict_pipeline.predict(pred_df)
        
        print("Predicted house price",results)
        
        return render_template('index.html',results=results)


if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True, port=8000)