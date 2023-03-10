import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle # will help to dump and load ML model

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')  # your default root page, it will open index.html by default 
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be like $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)