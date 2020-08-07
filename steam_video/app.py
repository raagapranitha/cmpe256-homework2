from flask import Flask, escape, request,jsonify
import json
from flask_jsonpify import jsonpify
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

app = Flask(__name__)


model_path = '/Users/raagapranithakolla/sjsu/cmpe256/homework2/cmpe256-homework2/steam_video/steam_video_NCF_8_[64, 32, 16, 8].h5'
print('using model: %s' % model_path)
model = load_model(model_path)
print('Loaded model!')

df = pd.read_json('/Users/raagapranithakolla/sjsu/cmpe256/homework2/cmpe256-homework2/steam_video/origin_steam_video_df.json')

mlp_bundle_embedding_weights = (next(iter(filter(lambda x: x.name == 'mlp_bundle_embedding', model.layers))).get_weights())

@app.route('/',methods=["GET"])
def getIndex():
	return jsonify('Welcome to Recommender System home page')

@app.route('/<int:bund_id>',methods=["GET"])
def getRecomm(bund_id):
	# this is a sample bundle to evaluate
	desired_bundle_id = 200
	model_path = 'steam_video_NCF_8_[64, 32, 16, 8].h5'
	print('using model: %s' % model_path)
	model = load_model(model_path)
	print('Loaded model!')

	mlp_bundle_embedding_weights = (next(iter(filter(lambda x: x.name == 'mlp_bundle_embedding', model.layers))).get_weights())

	# get the latent embedding for your desired bundle
	bundle_latent_matrix = mlp_bundle_embedding_weights[0]
	one_bundle_vector = bundle_latent_matrix[desired_bundle_id,:]
	one_bundle_vector = np.reshape(one_bundle_vector, (1,32))

	print('\nPerforming kmeans to find the nearest bundles...')
	# get 50 similar bundles
	# kmeans = KMeans(n_clusters=50, random_state=0, verbose=1).fit(bundle_latent_matrix)
	kmeans = MiniBatchKMeans(n_clusters=50, random_state=0, verbose=1).fit(bundle_latent_matrix)
	desired_bundle_label = kmeans.predict(one_bundle_vector)
	bundle_label = kmeans.labels_
	neighbors = []
	for bundle_id, bundle_label in enumerate(bundle_label):
	    # print('bundle_id:{0} bundle_label:{1}'.format(bundle_id, bundle_label))
	    if bundle_label == desired_bundle_label:
	        neighbors.append(bundle_id)
	print('Found {0} neighbor bundles/games.'.format(len(neighbors))) 

	# get the games in similar bundles' items
	games = []
	for bundle_id in neighbors:
	    games += list(df[df['bundleId'] == int(bundle_id)]['itemId'])
	print('Found {0} neighbor items from these games.'.format(len(games))) 

	games_arr = np.full(len(games), desired_bundle_id, dtype='int32')
	bundles = np.array(games, dtype='int32')

	print('\nRanking most likely games using the NeuMF model...')
	# and predict games for my bundle
	results = model.predict([games_arr,bundles],batch_size=100, verbose=0) 
	results = results.tolist()
	print('Ranked the games!')

	results_df = pd.DataFrame(np.nan, index=range(len(results)), columns=['probability','item_name', 'genre'])
	print(results_df.shape)

	# loop through and get the probability (of being in the bundle according to my model), the game, and the genre  
	for i, prob in enumerate(results):
	    # print('i: {0} prob: {1}'.format(i,prob))
	    results_df.loc[i] = [prob[0], df[df['itemId'] == i].iloc[0]['item_name'], df[df['itemId'] == i].iloc[0]['genre']]
	results_df = results_df.sort_values(by=['probability'], ascending=False)

	display_df = results_df.head(3)
	df_list = display_df.values.tolist()
	JSONP_data = jsonpify(df_list)
	return JSONP_data
	# return jsonify(results_df.head(10))

if __name__=="__main__":
	app.run(Debug=True,port=8080)

