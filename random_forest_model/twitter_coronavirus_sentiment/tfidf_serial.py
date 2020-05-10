import numpy as np 
import pandas as pd 
import re
import time
from collections import Counter
import csv


start_time = time.time()

# read in processed features corpus data file
processed_features = []
with open("processed_features_corpus.txt", 'r') as output:
    for row in output.readlines():
        processed_features.append(row)

# read in doc_freq from previous model
doc_freq = pd.read_csv("doc_freq.csv")
col_names = doc_freq.columns
doc_freq = doc_freq.rename(columns = {col_names[0]:'word', col_names[1]:'freq'})
doc_freq_dict = doc_freq.set_index('word').T.to_dict()


# get the document frequency from this text file
corpus = []

for line in processed_features:
    words = re.split(r'\W+', line)
    for word in words:
        corpus.append(word)

all_words_counter = Counter(corpus)
doc_freq_curr_dict = {}
count = N = len(processed_features)

for key, value in all_words_counter.items():
    # When building the vocabulary ignore terms that have a document frequency (absolute counts)
    # strictly lower the given threshold 7 (corpus-specific stop words).
    # When building the vocabulary ignore terms that have a document frequency (absolute counts)
    # strictly higher the given threshold 0.8 (corpus-specific stop words).
    if value > 7 and value < count * 0.8:
        doc_freq_curr_dict.update({key: value})




# save the document frequency dictionary as a csv file
with open('doc_freq_curr.csv', 'w') as f:
    w = csv.writer(f)
    w.writerows(doc_freq_curr_dict.items())


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
            tf, df = word_counter.get(feature), doc_freq_curr_dict.get(feature).get('freq')
            idf = np.log((N+1)/(df+1))+1
            result.append(tf*idf)
        else:
            result.append(0)
    return result

tfidf_matrix = []
for text in docs['text']:
    tfidf_matrix.append(get_tfidf(text))

processed_features_tfidf = np.array(tfidf_matrix)


# save tfidf_matrix to csv file
with open("tfidf_matrix.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(tfidf_matrix)



print("--- %s seconds ---" % (time.time() - start_time))

