# -*- coding: utf-8 -*-
import numpy as np
from flask import Flask, request, render_template
import joblib
app = Flask(__name__)
model = joblib.load("Stock DT","r")
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    Present_Price=float(request.form['Present_Price'])
    final_features = np.array(Present_Price).reshape(-1,1)
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Stock Price is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
