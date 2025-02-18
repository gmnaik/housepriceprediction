import pickle
from flask import Flask, request, render_template,jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS,cross_origin
#from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

#Route for  a home page
@app.route('/')
def index():
    
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)