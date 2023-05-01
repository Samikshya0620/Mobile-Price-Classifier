from click import style
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("form.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    battery_pow = int(request.form['batteryPower'])
    dual_sim = int(request.form['dualSim'])
    int_mem = int(request.form['intMemory'])
    px_height = int(request.form['pxHeight'])
    px_width = int(request.form['pxWidth'])
    RAM = int(request.form['ram'])
    four_g = int(request.form['fourG'])
    screen_height = int(request.form['scH'])
    screen_width = int(request.form['scW'])
    wifi = int(request.form['wifi'])

    data = {
        'battery_power' : [battery_pow],
        'int_memory' : [int_mem],
        'px_height' : [px_height],
        'px_width' : [px_width],
        'ram' : [RAM],
        'sc_h' : [screen_height],
        'sc_w' : [screen_width],
        'dual_sim' : [dual_sim],
        'four_g': [four_g],
        'wifi' : [wifi],
    }
    df = pd.DataFrame(data)
    scaler = joblib.load('scaler.pkl')
    # scaler = StandardScaler()
    num_features = ['battery_power','int_memory','px_height','px_width','ram','sc_h','sc_w']
    cat_features = ['dual_sim','four_g','wifi']
    final_df = df[num_features]
    scaler.transform(final_df)
    # print(final_df)
    X_scaled_list = scaler.transform(final_df)
    # print(X)
    num_data = np.array(X_scaled_list)
    print(num_data)
    cat_data = np.array(df[cat_features].values.tolist())
    X_scaled = np.concatenate((num_data,cat_data), axis=1)
    X_scaled = X_scaled.tolist()
    print(X_scaled)
    print(RAM)
    model = joblib.load("model.pkl")

    Y_pred = (model.predict(X_scaled))
    print(Y_pred)

    return render_template("slider.html", my_prediction = Y_pred)


if __name__ == "__main__":
    app.run()
