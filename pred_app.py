# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 12:33:38 2020

@author: ASUS
"""

from flask import Flask,render_template,url_for
from flask import request
from pred_api import make_prediction
from flask import jsonify


app=Flask(__name__)


@app.route("/")
def home():
    return render_template('home.html')
    
@app.route('/predict',methods=['Post'])
def predict():
    if request.method == 'POST':
        message=request.form['message']
        data=str(message)
        comment,my_prediction=make_prediction(data)
    return render_template('home.html',comm_in=comment,prediction=my_prediction)




if __name__=="__main__":
    app.run(debug=True)
    
