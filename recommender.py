'''
File with function for the Movie Recommender
'''

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF

def create_user_item_matrix(file_ratings, file_movies, n_rows):
    '''
    Create a user_item_matrix as pd.DataFrame.

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

    ratings = pd.read_csv(file_ratings, nrows=n_rows) #dataset to big --> kernel crashs everytime just use first 1_000_000 data

    #make a dict with movieId and title
    movies = pd.read_csv(file_movies, index_col=0)
    movies.drop(columns='genres', inplace=True)

    df_ratings_movie = pd.merge(ratings, movies, how='inner', on='movieId')
    df_ratings_movie.drop(columns=['timestamp','movieId'], inplace=True)
    
    return df_ratings_movie

def model_recommender(df):
    '''
    Uses pd.Dataframe of user-item-matrix and cerates and trains a NMF model.
    Creates user-item-matrix(R), user feature matrix (P) and item feature matrix(Q) .

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
    nmf: NMF(l1_ratio=0.5, max_iter=5000, n_components=150)
        The trained nmf model
    '''
    #Create user-item-matrix
    R = df.pivot(index='userId',
            columns='title',
            values='rating'
    )
    # Fill empty values
    R= R.fillna(2.5)

    #Create and train model
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




def user_recommendation(input_dict, R, Q, nmf):
    '''
    Uses trained model to make recommendations for new user.

    Paramters
    ---------
    input_dict: dict
            userinput with movies and ratings.
    R: pd.DataFrame
        user-item-matrix
    Q: pd.DataFrame
        item feature matrix
    nmf: NMF(l1_ratio=0.5, max_iter=5000, n_components=150)
        The trained model

    Returns
    ---------
    recommendations_user[:5]: list with first 5 entries
        The first 5 recommendations for the new_user.
    '''
    #create a sorted ranking list (first item --> first ranking corresponds to first movie)
    ranking = []
    for i in list(range(0,5)):
        ranking.append(input_dict[sorted(input_dict.keys())[i]])

    #create a sorted movie titel list (first item --> first movie)
    titel = []
    for i in list(range(5,10)):
        titel.append(input_dict[sorted(input_dict.keys())[i]])

    #create a dict out of ranking & titel list to use as input to create a pd.DataFrame(new_user)
    dict_user = {titel[i]:ranking[i] for i in range(len(titel))}
    new_user = pd.DataFrame(data=dict_user, index=['new_user'], columns=R.columns)
    new_user = new_user.fillna(2.54)

    #transform P matrix
    user_P = nmf.transform(new_user)

    #create user-item-matrix for new_user
    user_R = pd.DataFrame(np.dot(user_P, Q), columns=R.columns, index=['new_user'])

    #make recommendations and leave movie titels the new user used as input out
    recommendations = user_R.drop(columns=titel)

    #create a list with the recommendet movie titels
    recommendations_user = list(recommendations.sort_values(axis=1, by='new_user', ascending=False))

    return recommendations_user[:5]
