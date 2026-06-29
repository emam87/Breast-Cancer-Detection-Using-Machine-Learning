from flask import Flask, request, render_template
import pandas
import numpy as np
import pickle

from pyexpat import features

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
#flask app

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    featurs = request.form['feature']
    featurs_lst = featurs.split(',')

    # The model expects exactly 30 features. If the user accidentally includes the label '1' or '0' at the end, this will ignore it.
    np_features = np.array(featurs_lst[:30], dtype=float)

    # Scale the features using the trained StandardScaler
    np_features_scaled = scaler.transform(np_features.reshape(1, -1))

    pred = model.predict(np_features_scaled)

    output = "Cancerous" if pred[0]==1 else "Not Cancerous"

    return render_template('index.html', message=output)

#python main
if __name__ == "__main__":
    app.run(debug=True)





    #input
    #17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.0787, 1.095, 0.9053, 8.589, 153.4, 0.0064, 0.049, 0.0537, 0.0159, 0.03, 0.0062, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189, 1
    #20.57,17.77,132.9,1326,0.0847,0.0786,0.0869,0.0702,0.1812,0.0567,0.5435,0.7339,3.398,74.08,0.0052,0.0131,0.0186,0.0134,0.0139,0.0035,24.99,23.41,158.8,1956,0.1238,0.1866,0.2416,0.1860,0.2750,0.0890,0
    #13.54,14.36,87.46,566.3,0.0978,0.0813,0.0666,0.0478,0.1885,0.0577,0.2699,0.7886,2.058,23.56,0.0085,0.0202,0.0233,0.0110,0.0187,0.0030,15.11,19.26,99.70,711.2,0.1440,0.1773,0.2390,0.1288,0.2977,0.0726,1
    #11.89,21.17,76.39,433.8,0.0977,0.0618,0.0201,0.0239,0.1590,0.0591,0.2196,1.1500,1.450,16.35,0.0072,0.0156,0.0105,0.0070,0.0190,0.0025,13.25,27.21,85.09,522.9,0.1426,0.2187,0.1164,0.0826,0.2784,0.0800,0
    #15.46,19.48,101.7,748.9,0.1092,0.1223,0.1466,0.0809,0.1931,0.0579,0.4743,0.7859,3.094,48.31,0.0062,0.0406,0.0383,0.0158,0.0218,0.0033,19.26,26.00,124.9,1156,0.1546,0.2394,0.3791,0.1514,0.2837,0.0802,1
    #12.45,15.70,82.57,477.1,0.1278,0.1700,0.1578,0.0809,0.2087,0.0761,0.3345,0.8900,2.217,27.19,0.0075,0.0345,0.0350,0.0123,0.0200,0.0042,15.47,20.96,97.82,630.5,0.1530,0.3500,0.4000,0.1800,0.3100,0.0950,0