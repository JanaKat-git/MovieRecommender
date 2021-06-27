'''
File with function for the Movie Recommender
'''

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF

def create_user_item_matrix(file_ratings, file_movies, n_rows):
    '''
    Create a user-item matrix as pd.DataFrame.

    Paramters
    ---------
    file_ratings: str
        csv-file with ratings data
    file_movies: str
        csv-file with movie data
    n_rows: int
        Number of Rows reading from the csv-files.

    Returns
    ---------
    df_ratings_movie: pd.DataFrame
        The created user-item-matrix
    '''

    ratings = pd.read_csv(file_ratings, nrows=n_rows) 

    #make a dict with movieId and title
    movies = pd.read_csv(file_movies, index_col=0)
    movies.drop(columns='genres', inplace=True)

    df_ratings_movie = pd.merge(ratings, movies, how='inner', on='movieId')
    df_ratings_movie.drop(columns=['timestamp','movieId'], inplace=True)
    
    return df_ratings_movie


def model_recommender(df):
    '''
    Uses pd.Dataframe of the user-item matrix to create and train a NMF model.
    Creates a user-item matrix(R), user-feature matrix (P) and item-feature matrix(Q) .

    Paramters
    ---------
    df: pd.DataFrame
            dataframe of an user-item-matrix

    Returns
    ---------
    R: pd.DataFrame
        The created user-item-matrix
    P: pd.DataFrame
        The user feature matrix
    Q: pd.DataFrame
        item feature matrix
    nmf: nmf model
        The traines nmf model
    '''
    R = df.pivot(index='userId',
            columns='title',
            values='rating'
    )

    R= R.fillna(2.5)

    nmf = NMF(n_components = 150, 
            max_iter=5_000, 
            #alpha = 0.2, 
            l1_ratio= 0.5) # instantiate model
    nmf.fit(R) #fit R to the model

    #create Q: item feature matrix
    Q = pd.DataFrame(nmf.components_, columns=R.columns)

    #create P: user feature matrix
    P = pd.DataFrame(nmf.transform(R), index=R.index)

    #create R_hat: Matrixmultiplication of Q and P
    R_hat = pd.DataFrame(np.dot(P,Q), columns=R.columns, index=R.index)

    #evaluate error: delta(R, R_hat)
    nmf.reconstruction_err_

    return R, P, Q, nmf




def user_recommendation(input_dict, model_function):
    '''
    Uses trained model to make recommendations for new user.

    Paramters
    ---------
    input_dict: dict
            userinput with movies and ratings.
    model_function: function    
            Function to train a model and return user-item-matrix(R), user feature matrix (P) and model

    Returns
    ---------
    recommendations_user[:5]: list with first 5 entries
        The first 5 recommendations for the new_user.
    '''
    R, P, Q, nmf = model_function

    ranking = []
    for i in list(range(0,5)):
        ranking.append(input_dict[sorted(input_dict.keys())[i]])

    titel = []
    for i in list(range(5,10)):
        titel.append(input_dict[sorted(input_dict.keys())[i]])

    dict_user = {titel[i]:ranking[i] for i in range(len(titel))}
    
    new_user = pd.DataFrame(dict_user, index=['new_user'], columns=R.columns)
    new_user = new_user.fillna(2.54)

    user_P = nmf.transform(new_user)

    user_R = pd.DataFrame(np.dot(user_P, Q), columns=R.columns, index=['new_user'])

    recommendations = user_R.drop(columns=titel)

    recommendations_user = list(recommendations.sort_values(axis=1, by='new_user', ascending=False))

    return recommendations_user[:5]


