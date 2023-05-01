from click import style
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

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
        'dual_sim' : [dual_sim],
        'int_memory' : [int_mem],
        'px_height' : [px_height],
        'px_width' : [px_width],
        'ram' : [RAM],
        'four_g': [four_g],
        'sc_h' : [screen_height],
        'sc_w' : [screen_width],
        'wifi' : [wifi],
    }

    df = pd.DataFrame(data)
    print(df)
    model = joblib.load("model.pkl")

    Y_pred = (model.predict_proba(df))
    print(Y_pred)

    return render_template("slider.html", my_prediction = Y_pred)


if __name__ == "__main__":
    app.run(debug=True)
