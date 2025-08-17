from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model/rf_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        scores = [float(request.form["math"]),
                  float(request.form["reading"]),
                  float(request.form["writing"])]
        scores_scaled = scaler.transform([scores])
        prediction = model.predict(scores_scaled)[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)