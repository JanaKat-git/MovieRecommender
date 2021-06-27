'''
Controller file for the web appilaction
'''

from flask import Flask
from flask import render_template
from flask import request
from recommender import create_user_item_matrix, model_recommender, user_recommendation


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', title='Awesome Movie Recommender')

@app.route('/recommender')
def recommender():
    html_form_data = dict(request.args)
    print(html_form_data)

    df = create_user_item_matrix('ratings.csv', 'movies.csv', 10_000)
    recs= user_recommendation(html_form_data, model_recommender(df.iloc[0:500_000]))
    

    return render_template('recommendations.html',
                            movies = recs)

if __name__ == "__main__": 
    app.run(debug=True, port=5500) 

