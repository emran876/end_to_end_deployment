from flask import Flask, request
from tensorflow.keras.models import load_model
import pandas as pd
from flasgger import Swagger
import numpy as np


app = Flask(__name__)
Swagger(app)

#================================
# load the trained model
model = load_model("auto_mpg_model.h5")

# read data for scaling purpose
x_mean= np.zeros(6, dtype=float)
x_std = np.zeros(6, dtype=float)

with open('Xmean.npy', 'rb') as f:
    x_mean=np.load(f)
f.close()
with open('Xstd.npy', 'rb') as f:
    x_std=np.load(f)
f.close()
#===========================
# routines for scaler and encoder
def std_scaler(df, x_mean, x_std ):
    X_num = np.zeros((df.shape[0], 6), dtype=float)
    print(X_num.shape)
    for i in range(0, df.shape[0]):
        X_num[i,0] = ( df[i,0] - x_mean[0] ) / x_std[0]
        X_num[i,1] = ( df[i,1] - x_mean[1] ) / x_std[1]
        X_num[i,2] = ( df[i,2] - x_mean[2] ) / x_std[2]
        X_num[i,3] = ( df[i,3] - x_mean[3] ) / x_std[3]
        X_num[i,4] = ( df[i,4] - x_mean[4] ) / x_std[4]
        X_num[i,5] = ( df[i,5] - x_mean[5] ) / x_std[5]
    return X_num
    
def OneHot_Encoder(df):
    X_cat = np.zeros((df.shape[0], 3), dtype=float)
    for i in range(0, df.shape[0]):
        if (df[i] == 1):
            X_cat[i,0] = 1.0
            X_cat[i,1] = 0.0
            X_cat[i,2] = 0.0
        elif (df[i] == 2) :
            X_cat[i,0] = 0.0
            X_cat[i,1] = 1.0
            X_cat[i,2] = 0.0
        else :
            X_cat[i,0] = 0.0
            X_cat[i,1] = 0.0
            X_cat[i,2] = 1.0
    return X_cat
#========================================================

@app.route('/')
def welcome():
    return "Auto mpg model"
    
@app.route('/api', methods=["POST"])
def auto_mpg():
    """ Auto mpg
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
   
    # load data from file (input_data.csv)
    #data = request.files.get("file")
    
    df = pd.read_csv(request.files.get("file")) 
   
    df_num= df.iloc[:, 0:6].values
    df_cat= df.iloc[:, -1].values
    
    # encoding and scaling
    
    X_cat = np.zeros((df_cat.shape[0], 3), dtype=float)
    X_cat = OneHot_Encoder(df_cat)
    
    X_num = np.zeros((df_num.shape[0], 6), dtype=float)
    X_num= std_scaler(df_num, x_mean, x_std )
    
    
    X_norm=np.zeros((X_cat.shape[0], 9), dtype=float)
    
    # merge numerical and categorical data (column wise)
    X_norm= np.concatenate((X_num, X_cat), axis=1)
    
    y_pred = model.predict(X_norm)
    
    pred=float(y_pred[0])
    
    
    return str(pred)
      

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
