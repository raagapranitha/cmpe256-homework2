import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import scipy.sparse as sp
from keras import backend as K
from keras import initializers
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, Multiply, Reshape, Flatten, Concatenate
from keras.optimizers import Adam
from keras.regularizers import l2
import pandas as pd
import copy
from keras.utils import CustomObjectScope
import json, sys, random, os, datetime, math
print(tf.__version__)
print(keras.__version__)


# this is a nice rock/oldies playlist
desired_user_id = 500
model_path = '/Users/raagapranithakolla/sjsu/cmpe256/homework2/steam_video/steam_video_NCF_8_[64, 32, 16, 8].h5'
print('using model: %s' % model_path)
model = load_model(model_path)
print('Loaded model!')

df = pd.read_json('/Users/raagapranithakolla/sjsu/cmpe256/homework2/steam_video/origin_steam_video_df.json')

mlp_user_embedding_weights = (next(iter(filter(lambda x: x.name == 'mlp_user_embedding', model.layers))).get_weights())

# get the latent embedding for your desired user
user_latent_matrix = mlp_user_embedding_weights[0]
one_user_vector = user_latent_matrix[desired_user_id,:]
one_user_vector = np.reshape(one_user_vector, (1,32))

print('\nPerforming kmeans to find the nearest users/playlists...')
# get 100 similar users
# kmeans = KMeans(n_clusters=100, random_state=0, verbose=1).fit(user_latent_matrix)
kmeans = MiniBatchKMeans(n_clusters=100, random_state=0, verbose=1).fit(user_latent_matrix)
desired_user_label = kmeans.predict(one_user_vector)
user_label = kmeans.labels_
neighbors = []
for user_id, user_label in enumerate(user_label):
    print('user_id:{0} user_label:{1}'.format(user_id, user_label))
    if user_label == desired_user_label:
        neighbors.append(user_id)
print('Found {0} neighbor users/playlists.'.format(len(neighbors))) 

# get the tracks in similar users' playlists
games = []
for user_id in neighbors:
    games += list(df[df['bundleId'] == int(user_id)]['itemId'])
print('Found {0} neighbor items from these games.'.format(len(games))) 

users = np.full(len(games), desired_user_id, dtype='int32')
bundles = np.array(games, dtype='int32')

print('\nRanking most likely games using the NeuMF model...')
# and predict tracks for my user
results = model.predict([users,bundles],batch_size=100, verbose=0) 
results = results.tolist()
print('Ranked the games!')

results_df = pd.DataFrame(np.nan, index=range(len(results)), columns=['probability','item_name', 'genre'])
# print(results_df.shape)

# loop through and get the probability (of being in the bundle according to my model), the game, and the genre  
for i, prob in enumerate(results):
    # print('i: {0} prob: {1}'.format(i,prob))
    results_df.loc[i] = [prob[0], df[df['itemId'] == i].iloc[0]['item_name'], df[df['itemId'] == i].iloc[0]['genre']]
results_df = results_df.sort_values(by=['probability'], ascending=False)

print(results_df.head(10))


