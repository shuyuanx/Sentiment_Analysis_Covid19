import pickle
import multiprocessing as mp
from itertools import repeat
import pandas as pd
import numpy as np
import time
from textblob import TextBlob

start_time = time.time()

# number of processes
P = 4
# A thread pool of P processes
pool = mp.Pool(P)



# Load model from file
pkl_filename = "rfc_mp_mt.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

# read in processed features corpus data file
processed_features = []
with open("processed_features_corpus.txt", 'r') as output:
    for row in output.readlines():
        processed_features.append(row)

# tag sentiments for twitter text data
sentiments = []
for tweet_text in processed_features:
    sentiment = TextBlob(tweet_text).sentiment
    polarity = sentiment.polarity
    if polarity < 0:
        sentiments.append(0)
    elif polarity > 0:
        sentiments.append(1)
    else:
        sentiments.append(2)

# import tfidf data
tfidf_df = pd.read_csv("tfidf_matrix.csv",header=None)
tfidf_df['sentiment'] = sentiments
tfidf_data = np.array(tfidf_df)

# make predictions with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
	    if isinstance(node['left'], dict):
	        return predict(node['left'], row)
	    else:
	        return node['left']
    else:
	    if isinstance(node['right'], dict):
	        return predict(node['right'], row)
	    else:
	        return node['right']

# make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    #predictions = pool.starmap(predict, zip(trees,repeat(row)))
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


# make prediction for all observations
# negative: 0, positive: 1, neutral: 2
predicted = pool.starmap(bagging_predict, zip(repeat(model), tfidf_data))
#predicted = [bagging_predict(model, row) for row in tfidf_data]




print("--- %s seconds ---" % (time.time() - start_time))
