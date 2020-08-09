from flask import Flask, escape, request,jsonify, render_template,session,redirect
import json
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
import os

app = Flask(__name__)



def getRecomm(bund_id):
	desired_bundle_id = bund_id
	model_path = os.getcwd()+'/steam_video_NCF_8_[64, 32, 16, 8].h5'
	json_file_path = os.getcwd()+'/origin_steam_video_df.json'
	print('using model: %s' % model_path)
	model = load_model(model_path)
	print('Loaded model!')
	df = pd.read_json(json_file_path)
	# this is a sample bundle to evaluate
	mlp_bundle_embedding_weights = (next(iter(filter(lambda x: x.name == 'mlp_bundle_embedding', model.layers))).get_weights())
	# get the latent embedding for your desired bundle
	bundle_latent_matrix = mlp_bundle_embedding_weights[0]
	one_bundle_vector = bundle_latent_matrix[desired_bundle_id,:]
	one_bundle_vector = np.reshape(one_bundle_vector, (1,32))

	print('\nPerforming kmeans to find the nearest bundles...')
	# get 50 similar bundles
	kmeans = MiniBatchKMeans(n_clusters=50, random_state=0, verbose=1).fit(bundle_latent_matrix)
	desired_bundle_label = kmeans.predict(one_bundle_vector)
	bundle_label = kmeans.labels_
	neighbors = []
	for bundle_id, bundle_label in enumerate(bundle_label):
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

	# loop through and get the probability (of being in the bundle according to my model), the game, and the genre  
	for i, prob in enumerate(results):
	    results_df.loc[i] = [prob[0], df[df['itemId'] == i].iloc[0]['item_name'], df[df['itemId'] == i].iloc[0]['genre']]
	results_df = results_df.sort_values(by=['probability'], ascending=False)

	display_df = (results_df.head(3))
	return display_df

@app.route('/',methods=["GET","POST"])
def getIndex():
	errors = []
	results = {}
	if request.method == "POST":
        # get bundleId that the user has entered
		try:
			bundle_id = request.form['bundleId']
			df_res= getRecomm(int(bundle_id))
			return render_template('simple.html', tables=[df_res.to_html(classes='data')], titles=df_res.columns.values)
		except:
			errors.append("Unable to get URL. Please make sure it's valid and try again.")
	else:

		return render_template('index.html',error=errors,results=results)


if __name__=="__main__":
	app.run(Debug=True,port=8080)

