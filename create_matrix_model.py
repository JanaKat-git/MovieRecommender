'''
File to:
(1) load the user-item matrix and 
(2) train an NMF model (use recommender.model_recommender)
(3) pickle the model and matrices in the file 'nmf_model'
'''

from recommender import  model_recommender, create_user_item_matrix
import pandas as pd
import pickle

df = create_user_item_matrix('RATINGS', 'MOVIES', 10_000)
df.to_csv('user_item_matrix.csv')

#Load the user_item_matrix
df = pd.read_csv('user_item_matrix.csv')

#Create and train a NMF model with the user_item_matrix
R, P, Q, nmf = model_recommender(df)


#Pickle the model and matrices
with open("nmf_m.pkl", "wb") as f:
    pickle.dump([R, P, Q, nmf], f)






