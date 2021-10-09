import pickle

import numpy as np
import sklearn
from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('RentPredictionRegressionModel.pkl', 'rb'))
rent = pd.read_csv("Rent_Mumbai.csv")


@app.route('/')
def index():
    locality = sorted(rent['Locality'].unique())
    bhk = rent['Type'].unique()
    furnishing = sorted(rent['Furnishing'].unique())
    bathrooms = sorted(rent['Bathrooms'].unique())
    balcony = sorted(rent['Balcony'].unique())
    parking = sorted(rent['Parking'].unique())

    return render_template('index.html', locality=locality, bhk=bhk, furnishing=furnishing, bathrooms=bathrooms,
                           balcony=balcony, parking=parking)


@app.route('/predict', methods=['POST'])
def predict():
    locality = request.form.get('locality')
    bhk = request.form.get('bhk')
    furnishing = request.form.get('furnishing')
    bathrooms = int(request.form.get('bathrooms'))
    balcony = int(request.form.get('balcony'))
    parking = int(request.form.get('parking'))
    buildup = int(request.form.get('buildup'))
    carpet = int(request.form.get('carpet'))

    pred = model.predict(pd.DataFrame([[locality, bhk, buildup, furnishing, bathrooms, balcony, parking, carpet]],columns=['Locality', 'Type', 'Build_up_area(sq.ft)', 'Furnishing', 'Bathrooms', 'Balcony', 'Parking', 'Carpet_area(sq.ft)']))
    return str(np.round(pred[0], 2))


if __name__ == "__main__":
    app.run(debug=True)
