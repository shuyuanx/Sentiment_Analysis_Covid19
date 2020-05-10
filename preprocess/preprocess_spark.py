from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext 
from pyspark.sql.functions import udf, lit, UserDefinedFunction
from pyspark.sql.types import *

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

conf = SparkConf().setMaster('local').setAppName('Preprocess')
sc = SparkContext(conf = conf)
sqlc = SQLContext(sc) 

tweets = pd.read_csv('corona.csv', skiprows=[43001, 57301, 71601, 85901, 100201])
tweetsdf = sqlc.createDataFrame(tweets.astype(str))

def tokenize(col):
	lower = col.lower()
	return nltk.word_tokenize(lower)

tokenize_udf = UserDefinedFunction(tokenize, StringType())
# tweetsdf.withColumn('words', lit(0))
tweetsdf=tweetsdf.withColumn('words', tokenize_udf("text"))

stemming = PorterStemmer()
def stem_list(col):
    # my_list = row['words']
    stemmed_list = [stemming.stem(word) for word in col]
    return (stemmed_list)

stem_udf = udf(stem_list)
tweetsdf=tweetsdf.withColumn('stemmed_words', stem_udf("words"))

stop_words=set(stopwords.words('english'))
def remove_stops(row):
    # my_list = row['stemmed_words']
    meaningful_words = [w for w in row if not w in stop_words]
    return (meaningful_words)

stop_udf = udf(remove_stops)
tweetsdf=tweetsdf.withColumn('stem_meaningful', stop_udf("stemmed_words"))

def rejoin_words(row):
    # my_list = row['stem_meaningful']
    joined_words = ( " ".join(row))
    return joined_words

rejoin_udf = udf(rejoin_words)
tweetsdf=tweetsdf.withColumn('processed', rejoin_udf("stem_meaningful"))
# tweetsdf.filter(lambda word : word[0] not in stop_words)

def tag(row):
	# my_list = row['stem_meaningful']
	tagged = nltk.pos_tag(row)
	return tagged

tag_udf = udf(tag)
tweetsdf=tweetsdf.withColumn('tagged', tag_udf("processed"))

cols_to_drop = ['words', 'stemmed_words', 'stem_meaningful']
resultdf = tweetsdf.drop(*cols_to_drop)

resultdf.write.csv('corona_processed.csv')