import numpy as np
from flask import Flask,request,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('model3.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['Post'])
def predict():
    int_features=[int(x) for x in request.form.values()]

    final_features=[np.array(int_features)]

    prediction=model.predict(final_features)

    return render_template('index.html', prediction_text='Rent has to be around Rs. {}'.format(round((prediction[0]),2)))

if __name__ == "__main__":
    app.run(debug=True)
