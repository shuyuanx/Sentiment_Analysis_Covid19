import numpy as np 
import pandas as pd 
import re
import nltk 
import matplotlib.pyplot as plt
import sqlite3
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Read sqlite query results into a pandas DataFrame
con = sqlite3.connect("database.sqlite")
airline_tweets = pd.read_sql_query("SELECT * from Tweets", con)

# distribution of sentiment for each individual airline
airline_sentiment = airline_tweets.groupby(['airline', 'airline_sentiment']).airline_sentiment.count().unstack()

# data cleaning
features = airline_tweets['text'].values
labels = airline_tweets['airline_sentiment'].values

processed_features = []

for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

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

with open("processed_features_corpus.txt", 'w') as output:
    for row in processed_features:
        output.write(str(row) + '\n')

with open("labels.txt", 'w') as output:
    for label in labels:
        output.write(str(label) + '\n')

con.close()
