import pickle
from flask import Flask,request
import numpy as np
import os

with open(os.path.join('artifacts','model.pkl'),'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return f"Welcome to Docker basics"

@app.route('/predict',methods=['POST','GET'])
def prediction():
    s_length = request.args.get('s_length')
    s_width = request.args.get('s_width')
    p_length = request.args.get('p_length')
    p_width = request.args.get('p_width')

    prediction = model.predict(np.array([[s_length,s_width,p_length,p_width]]))

    return str(prediction)


if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=5005)
