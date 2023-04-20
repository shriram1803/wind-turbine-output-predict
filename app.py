import flask
from flask import request, jsonify
from flask_cors import CORS
import pandas as pd
from xgboost import XGBRegressor
import requests
import json
import tensorflow as tf
import numpy
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy import array
from datetime import datetime, timedelta

app = flask.Flask(__name__, static_url_path='')
CORS(app)
# wind forecast need to be written
def predictForWindSpeedAndWindDirection(ws,wd,name=''):
    X = [[ws,wd]]
    xgr=XGBRegressor()
    df = pd.DataFrame(X, columns=['WindSpeed(m/s)','WindDirection'])
    xgr.load_model('models/test_model.bin')
    result = xgr.predict(df)[0]
    return result; 

def parseweatherapi(apikey,lat,lng):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "lat=" + str(lat) + "&lon=" + str(lng) +"&appid=" + apikey
    response = requests.get(complete_url)
    x = response.json()
    print(x)
    if x["cod"] != "404":      
        y = x["wind"]       
        ws = y["speed"]         
        wd = y["deg"]    
        name = x["name"]
        return name,ws,wd;

def getForecastData(apikey,lat,lng):
    base_url = "http://api.openweathermap.org/data/2.5/forecast?"
    complete_url = base_url + "lat=" + str(lat) + "&lon=" + str(lng) +"&appid=" + apikey
    response = requests.get(complete_url)  
    data = json.loads(response.text)
    
    cnt = data["cnt"]
    newdata = {}
    time=[]
    ws=[]
    wd=[]
    for item in data["list"]:
        time.append(item["dt_txt"])
        ws.append(item["wind"]["speed"])
        wd.append(item["wind"]["deg"])
    newdata["time"]=time
    newdata["ws"]=ws
    newdata["wd"]=wd
    def predict(x_input,temp_input,model):
        lst_output=[]
        n_steps=10
        i=0
        while(i<10):    
            if(len(temp_input)>10):
                x_input=np.array(temp_input[1:])
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i=i+1  
        lst_output=scaler.inverse_transform(lst_output) 
        return lst_output
    df = pd.DataFrame.from_dict(newdata)
    wsdf=df.reset_index()['ws']
    scaler=MinMaxScaler(feature_range=(0,1))
    wsdf=scaler.fit_transform(np.array(wsdf).reshape(-1,1))
    wsmodel = tf.keras.models.load_model('models/windspeed100.h5')
    ws_x_input=wsdf[len(wsdf)-10:].reshape(1,-1)
    ws_temp_input=list(ws_x_input)
    ws_temp_input=ws_temp_input[0].tolist()
    ws_prediction=predict(ws_x_input, ws_temp_input, wsmodel)
    wddf=df['wd']
    wddf=scaler.fit_transform(np.array(wddf).reshape(-1,1))
    wdmodel = tf.keras.models.load_model('models/winddirection100.h5')
    wd_x_input=wddf[len(wddf)-10:].reshape(1,-1)
    wd_temp_input=list(wd_x_input)
    wd_temp_input=wd_temp_input[0].tolist()
    wd_prediction = predict(wd_x_input, wd_temp_input, wdmodel)
    for i in range(10):
        curdate = newdata["time"][-1]
        newdate=(datetime.strptime(curdate,"%Y-%m-%d %H:%M:%S")+timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
        newdata["time"].append(newdate)
        newdata["ws"].extend(ws_prediction[i])
        newdata["wd"].extend(wd_prediction[i])
    return newdata
        
@app.route('/')
def hello_world():
    return 'This is my first API call!'


@app.route('/predict-ws-and-wd',methods=['POST','GET'])
def predictwswd():
    if request.method =='POST':  
        data = request.get_json()     
        ws = float(data['ws'])
        wd = float(data['wd'])
        try:
           result= predictForWindSpeedAndWindDirection(ws, wd)
           return jsonify({'type':'windspeed and winddirection','result':str(result)})
        except Exception as e:
            print(e)
            return "error"

@app.route('/predict-lat-and-lng',methods=['POST','GET'])
def predictlatlng():
    if request.method =='POST':  

        data = request.get_json()     
        lat = float(data['lat'])
        lng = float(data['lng'])
        apikey = str(data['apikey'])
        try:
            name,ws,wd = parseweatherapi(apikey, lat, lng)
            result=predictForWindSpeedAndWindDirection(ws, wd)
            return jsonify({'type':'lat and lng','result':str(result),"ws":str(ws),"wd":str(wd)})
        except Exception as e:
            print(e)
            return "error"



@app.route('/forecast-6-days',methods=['POST','GET'])
def forecast6days():
    if request.method =='POST':  
        data = request.get_json()     
        lat = float(data['lat'])
        lng = float(data['lng'])
        apikey = str(data['apikey'])
        try:
           return jsonify({"data":getForecastData(apikey, lat, lng)})
        except Exception as e:
            print(e)
            return "error"

@app.route('/forecast-and-predict-6-days',methods=['POST','GET'])
def forecastAndPredict6days():
    if request.method =='POST':  
        data = request.get_json()     
        lat = float(data['lat'])
        lng = float(data['lng'])
        apikey = str(data['apikey']);
        try:
           data= getForecastData(apikey, lat, lng)
           data["wp"]=[]
           for i in range(len(data["time"])):
            result=predictForWindSpeedAndWindDirection(data["ws"][i], int(data["wd"][i]))
            data["wp"].append(str(result))
           return jsonify({"data":data});
        except Exception as e:
            print(e)
            return "error"
        



from flask_pymongo import pymongo
import json

from dotenv import load_dotenv
import os

# load environment variables from .env file
load_dotenv()

# use environment variables
MONGO = os.getenv('MONGO_URI')

# Set up MongoDB connection
client = pymongo.MongoClient(MONGO)
db = client["awtproj"]
collection = db["coordinates"]

# API to create the `coordinates` schema with a single document
@app.route('/api/create_coordinates', methods=['POST'])
def create_coordinates():
    # Create the `coordinates` schema with a single document
    print("create_called")
    result = collection.insert_one({'lat': "0", 'lng': "0"})
    
    # Return the ID of the inserted document as a JSON response
    return json.dumps({'success': True, 'id': str(result.inserted_id)}), 201

# API to update the `lat` and `lng` values of the single document
@app.route('/api/update_coordinates', methods=['POST'])
def update_coordinates():
    # Get the request data as a JSON object
    print("update_called")
    data = request.get_json()
    
    # Update the `lat` and `lng` values of the single document
    result = collection.update_one({}, {'$set': {'lat': data['lat'], 'lng': data['lng']}})
    
    # Return a success message as a JSON response
    return json.dumps({'success': True}), 200






if __name__ == '__main__':
    app.run(debug=False)