'''
Controller file for the web appilaction

The central file of the application
'''

from flask import Flask
from flask import render_template
from flask import request
import pandas as pd
from recommender import  user_recommendation
import pickle



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', title='Awesome Movie Recommender')

@app.route('/recommender')
def recommender():
    #save user input as dict and print it
    html_from_data = dict(request.args)
    print(html_from_data)

    #load pickled model
    with open("nmf_m.pkl", "rb") as f:
        R, P, Q, nmf = pickle.load(f) 

    
    #make recommendations for new user
    recs= user_recommendation(html_from_data, R, Q, nmf)
    print(recs)
    
    return render_template("recommendations.html", movies=recs)

if __name__ == "__main__": 
    app.run(debug=True, port=5500) 

