import numpy as np
import pandas as pd
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# read the file
tweets = pd.read_csv('corona.csv', skiprows=[43001, 57301, 71601, 85901, 100201])

# conver to lower-case alphabet
tweets['text'] = tweets['text'].str.lower()
# tokenize
def identify_tokens(row):
    review = row['text']
    tokens = nltk.word_tokenize(review)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words
tweets['words'] = tweets.apply(identify_tokens, axis=1)

# stemming
stemming = PorterStemmer()
def stem_list(row):
    my_list = row['words']
    stemmed_list = [stemming.stem(word) for word in my_list]
    return (stemmed_list)

tweets['stemmed_words'] = tweets.apply(stem_list, axis=1)

# remove stop words
stop_words=set(stopwords.words('english'))
def remove_stops(row):
    my_list = row['stemmed_words']
    meaningful_words = [w for w in my_list if not w in stop_words]
    return (meaningful_words)

tweets['stem_meaningful'] = tweets.apply(remove_stops, axis=1)

def rejoin_words(row):
    my_list = row['stem_meaningful']
    joined_words = ( " ".join(my_list))
    return joined_words

tweets['processed'] = tweets.apply(rejoin_words, axis=1)

def tag(row):
	my_list = row['stem_meaningful']
	tagged = nltk.pos_tag(my_list)
	return tagged
tweets['tagged'] = tweets.apply(tag, axis=1)

cols_to_drop = ['words', 'stemmed_words', 'stem_meaningful']
tweets.drop(cols_to_drop, inplace=True, axis=1)

tweets.to_csv(r'corona_processed.csv')