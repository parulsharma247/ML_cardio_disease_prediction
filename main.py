import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load(open('trained_model_joblib.pkl', 'rb'))

@app.route("/", methods=['GET'])
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    print(int_features)
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    op = round(prediction[0], 2)
    if op == 0:
        output = "Less"
    else:
        output = "High"

    return render_template('result.html', prediction_text='Your Cardiac Disease Probability is {}'.format(output))



if __name__ == "__main__":
    app.run(port=5000, debug=True)