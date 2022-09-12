from flask import Flask, render_template, request
import pickle 
import numpy as np
import pandas as pd

model = pickle.load(open('brt_can.pkl','rb'))

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('Index1.html')


@app.route('/Breast-Cancer')
def BC():
    return render_template('index.html')

@app.route('/Predict', methods = ['POST'])
def home():
    
    input_features = [float(x) for x in request.form.values()]
    features_values = [np.array(input_features)]
    print(features_values)
    #data1 = request.form['a']
    #data2 = request.form['b']
    #data3 = request.form['c']
    #data4 = request.form['d']
    
    features_name = ['a','b', 'c', 'd', 'e']
    
    #data5 = request.form['e']
    #print(data1)
    #arr = np.array([[data1, data2, data3, data4, data5]])
    df = pd.DataFrame(features_values, columns = features_name)
    print(df)
    pred = model.predict(features_values)
    return render_template('Predict.html',data = pred)


@app.route('/Stroke Prediction')
def SP():
    return render_template('index.html')
    
    
    

if __name__ == "__main__":
    app.run(debug = True)