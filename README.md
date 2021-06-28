# MovieRecommender with Web Interface

## Description
Building a movie **Recommender** with **web interface**. Derive a **user-item matrix** (using the data set from https://grouplens.org/datasets/movielens/), 
train and pickle a NMF model. Use this model to make recommendations from a **user input** on web interface.


## Usage
1) Create an user-item matrix using a movie and a ratings (0 to 5) table and store it as csv file. 
    - I used the data set from: https://grouplens.org/datasets/movielens/
    - Use `create_matrix_model.py` line 12 and include the ratings.csv and movie.csv here
2) create and train a NMF model and pickle model with `create_matrix_model.py`

### Run local:
***Notes***: this repo requires you to have python and scikit-learn (0.24.2) running on your machine

1) Run `application.py`
2) Open link in Browser
3) Use trained model to make recommendations for user (input in web interface)




