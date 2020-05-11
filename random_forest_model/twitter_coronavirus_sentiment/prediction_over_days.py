#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 20:34:00 2020

@author: weiruchen
"""

import pickle
import multiprocessing as mp
from itertools import repeat
import pandas as pd
import numpy as np
from textblob import TextBlob
import re
from collections import Counter



# import twitter data related to coronavirus
coronavirus_sentiment = pd.read_csv("corona_processed2.csv")[['date','processed']]


for i in range(len(coronavirus_sentiment)):
    coronavirus_sentiment.iloc[i]['date'] = str(coronavirus_sentiment.iloc[i]['date'])[0:10]

coronavirus_sentiment = coronavirus_sentiment[coronavirus_sentiment.date != 'date']

# extracting text data
text_data = []
for date, df_date in coronavirus_sentiment.groupby('date'):
    text_data.append(list(df_date['processed']))


print(set(coronavirus_sentiment.date.values))

# data cleaning
def data_cleaning(text_data):
    
    processed_features = []
    
    for sentence in range(0, len(text_data)):
        # Remove all the special characters
        processed_feature = re.sub(r'\W', ' ', str(text_data[sentence]))
    
        # remove all single characters
        processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    
        # Remove single characters from the start
        processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 
    
        # Substituting multiple spaces with single space
        processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    
        # Removing prefixed 'b'
        processed_feature = re.sub(r'^b\s+', '', processed_feature)
    
        # Converting to Lowercase
        processed_feature = processed_feature.lower()
    
        processed_features.append(processed_feature)
    
    return processed_features



processed_feature = []
for text_date in text_data:
    processed_feature.append(data_cleaning(text_date))


# read in doc_freq from previous model
doc_freq = pd.read_csv("doc_freq.csv")
col_names = doc_freq.columns
doc_freq = doc_freq.rename(columns = {col_names[0]:'word', col_names[1]:'freq'})
doc_freq_dict = doc_freq.set_index('word').T.to_dict()


# get the document frequency from this text file

def get_doc_freq_curr(processed_features):

    corpus = []

    for line in processed_features:
        words = re.split(r'\W+', line)
        for word in words:
            corpus.append(word)
            
    all_words_counter = Counter(corpus)
    doc_freq_curr_dict = {}
    count = len(processed_features)
    
    for key, value in all_words_counter.items():
        # When building the vocabulary ignore terms that have a document frequency (absolute counts)
        # strictly lower the given threshold 7 (corpus-specific stop words).
        # When building the vocabulary ignore terms that have a document frequency (absolute counts)
        # strictly higher the given threshold 0.8 (corpus-specific stop words).
        if value > 7 and value < count * 0.8:
            doc_freq_curr_dict.update({key: value})
    return doc_freq_curr_dict

"""
doc_freq_curr_dict = []
for feature in processed_feature:
    doc_freq_curr_dict.append(get_doc_freq_curr(feature))
"""

processed_features = []
for feature in processed_feature:
    processed_features += feature

doc_freq_curr_dict = get_doc_freq_curr(processed_features)
doc_keys = doc_freq_curr_dict.keys()





def get_tfidf(doc_text):
    words = re.split(r'\W+', doc_text)
    word_counter = Counter(words)
    keys = word_counter.keys()
    result = []
    for feature in features:
        if feature in keys and feature in doc_keys:
            tf, df = word_counter.get(feature), doc_freq_curr_dict.get(feature)
            idf = np.log((N+1)/(df+1))+1
            result.append(tf*idf)
        else:
            result.append(0)
    return result


tfidf_matrices = []
features = list(doc_freq_dict.keys())
for feature in processed_feature:
    docs = pd.DataFrame(feature)
    docs.columns = ["text"]
    
    N = len(docs)
    
    
    tfidf_matrix = []
    for text in docs['text']:
        tfidf_matrix.append(get_tfidf(text))
    
    processed_features_tfidf = np.array(tfidf_matrix)
    tfidf_matrices.append(processed_features_tfidf)






# number of processes
P = 4
# A thread pool of P processes
pool = mp.Pool(P)



# Load model from file
pkl_filename = "rfc_mp_mt.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)


# tag sentiments for twitter text data
sentiments = []
for feature in processed_feature:
    sentiment = []
    for tweet_text in feature:
        tweet_sentiment = TextBlob(tweet_text).sentiment
        polarity = tweet_sentiment.polarity
        if polarity < 0:
            sentiment.append(0)
        elif polarity > 0:
            sentiment.append(1)
        else:
            sentiment.append(2)
    sentiments.append(sentiment)

# import tfidf data

tfidf_df = []
tfidf_data = []
for i in range(len(tfidf_matrices)):
    tfidf_matrix = tfidf_matrices[i]
    df = pd.DataFrame(tfidf_matrix)
    df['sentiment'] = sentiments[i]
    tfidf_df.append(df)
    tfidf_data.append(np.array(df))




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
predictions = []
for data in tfidf_data:
    predicted = pool.starmap(bagging_predict, zip(repeat(model), data))
    predictions.append(predicted)
#predicted = [bagging_predict(model, row) for row in tfidf_data]

for prediction in predictions:
    print(Counter(prediction))
    




