from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name___)

model = pickle.load(open("model.pkl","rb"))


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    income = float(request.form["income"])
    score = float(request.form["score"])

    data = np.array([[income,score]])
    result = model.predict(data)[0]
 
    return render_template("result.html",prediction=result)

   