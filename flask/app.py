# from os import name

# from flask import Flask, redirect, render_template, url_for,request
# app = Flask(__name__)

# @app.route('/')
# def hello_world():
#     return 'Hello, World!'

# @app.route('/greet/<name>')
# def greet(name):
#     return f'Hello, {name}!'

# @app.route('/add')
# def add():
#     return redirect(url_for('hello_world'))

# @app.route('/<name>')
# def jumanji(name):
#     return render_template("index.html" , name=name)
    
# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         user = request.form["username"]
#         pwd = request.form["password"]
#         return redirect(url_for('user', urs=user, pwd=pwd))
#     else:
#         return render_template("login.html")

# @app.route('/greet/<urs>/<pwd>')
# def user(urs, pwd):
#     return render_template("urs.html", urs=urs, pwd=pwd)

# if __name__ == '__main__':    
#     app.run(debug=True)


from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # 1. Get data from form
        features = [
            float(request.form['Area']),
            float(request.form['Perimeter']),
            float(request.form['Major_Axis_Length']),
            float(request.form['Minor_Axis_Length']),
            float(request.form['Convex_Area']),
            float(request.form['Equiv_Diameter']),
            float(request.form['Eccentricity']),
            float(request.form['Solidity']),
            float(request.form['Extent']),
            float(request.form['Roundness']),
            float(request.form['Aspect_Ration']),
            float(request.form['Compactness'])  # <--- Added this!
        ]
        
        # 2. Update feature names list to match model exactly
        feature_names = [
            'Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length', 
            'Convex_Area', 'Equiv_Diameter', 'Eccentricity', 'Solidity', 
            'Extent', 'Roundness', 'Aspect_Ration', 'Compactness'
        ]
                         
        # 3. Create DataFrame
        df = pd.DataFrame([features], columns=feature_names)
        
        # 4. Predict
        prediction = model.predict(df)[0]
        
        return f"""
        <div style="text-align:center; padding: 50px;">
            <h1 style="color: #d35400;">Prediction: {prediction}</h1>
            <a href='/predict' style="background: #27ae60; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Try Again</a>
        </div>
        """

if __name__ == '__main__':
    app.run(debug=True)