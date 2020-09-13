#import libraries
import numpy as np
from flask import Flask,render_template,request

import pickle

app = Flask(__name__)

model = pickle.load(open('model.pk1','rb'))

@app.route('/')
def home():
    return  render_template('index.html')

@app.route('/predict',methods= ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    predictions = model.predict(final_features)
    output = round(predictions[0],2)
    return  render_template('index.html',prediction_text = 'CO2 Emission of the vehicle is:{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
