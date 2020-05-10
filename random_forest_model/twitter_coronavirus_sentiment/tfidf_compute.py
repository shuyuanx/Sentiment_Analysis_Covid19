import numpy as np 
import pandas as pd 
#import csv
import re
from collections import Counter
import multiprocessing as mp
import time


start_time = time.time()

# read in processed features corpus data file
processed_features = []
with open("processed_features_corpus.txt", 'r') as output:
    for row in output.readlines():
        processed_features.append(row)



doc_freq = pd.read_csv("doc_freq.csv")
col_names = doc_freq.columns
doc_freq = doc_freq.rename(columns = {col_names[0]:'word', col_names[1]:'freq'})
doc_freq_dict = doc_freq.set_index('word').T.to_dict()

doc_freq_curr = pd.read_csv("doc_freq_curr.csv")
col_names = doc_freq_curr.columns
doc_freq_curr = doc_freq_curr.rename(columns = {col_names[0]:'word', col_names[1]:'freq'})
doc_freq_curr_dict = doc_freq_curr.set_index('word').T.to_dict()
doc_keys = doc_freq_curr_dict.keys()


#docs = pd.read_fwf('processed_features_corpus.txt', header=None)
docs = pd.DataFrame(processed_features)
docs.columns = ["text"]

features = list(doc_freq_dict.keys())
N = len(docs)


def get_tfidf(doc_text):
    words = re.split(r'\W+', doc_text)
    word_counter = Counter(words)
    keys = word_counter.keys()
    result = []
    for feature in features:
        if feature in keys and feature in doc_keys:
            tf, df = word_counter.get(feature), doc_freq_dict.get(feature).get('freq')
            idf = np.log((N+1)/(df+1))+1
            result.append(tf*idf)
        else:
            result.append(0)
    return result


# number of processes
P = 4
# A thread pool of P processes
pool = mp.Pool(P)

tfidf_matrix = pool.map(get_tfidf, [text for text in docs['text']])

"""
# save tfidf_matrix to csv file
with open("tfidf_matrix.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(tfidf_matrix)
"""
    
#df = pd.read_csv("tfidf_matrix.csv", header=None)

print("--- %s seconds ---" % (time.time() - start_time))


