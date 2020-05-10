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
import sys
import re
from collections import Counter

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

# generate word embeddings (convert text to numberical vectors)
# method1: bag of words (not used here)

# method2: TF-IDF
"""
vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features_tfidf = vectorizer.fit_transform(processed_features).toarray()
"""

# write our own tfidf algorithm
"""
with open("processed_features_corpus.txt", 'w') as output:
    for row in processed_features:
        output.write(str(row) + '\n')
"""

corpus = []

for line in processed_features:
    words = re.split(r'\W+', line)
    for word in words:
        corpus.append(word)

all_words_counter = Counter(corpus)
doc_freq_dict = {}
count = N = len(processed_features)

for key, value in all_words_counter.items():
    # When building the vocabulary ignore terms that have a document frequency (absolute counts)
    # strictly lower the given threshold 7 (corpus-specific stop words).
    # When building the vocabulary ignore terms that have a document frequency (absolute counts)
    # strictly higher the given threshold 0.8 (corpus-specific stop words).
    if value > 7 and value < count * 0.8:
        doc_freq_dict.update({key: value})

"""
# save the document frequency dictionary as a csv file
with open('doc_freq.csv', 'w') as f:
    w = csv.writer(f)
    w.writerows(doc_freq_dict.items())
"""

doc_freq = pd.read_csv("doc_freq.csv")
col_names = doc_freq.columns
doc_freq = doc_freq.rename(columns = {col_names[0]:'word', col_names[1]:'freq'})
doc_freq_dict = doc_freq.set_index('word').T.to_dict()

docs = pd.read_fwf('processed_features_corpus.txt', header=None)
docs.columns = ["text"]

features = list(doc_freq_dict.keys())
N = len(docs)

def get_tfidf(doc_text):
    words = re.split(r'\W+', doc_text)
    word_counter = Counter(words)
    keys = word_counter.keys()
    result = []
    for feature in features:
        if feature in keys:
            tf, df = word_counter.get(feature), doc_freq_dict.get(feature).get('freq')
            idf = np.log((N+1)/(df+1))+1
            result.append(tf*idf)
        else:
            result.append(0)
    return result

tfidf_matrix = []
for text in docs['text']:
    tfidf_matrix.append(get_tfidf(text))

processed_features_tfidf = np.array(tfidf_matrix)
"""
# save tfidf_matrix to csv file
with open("tfidf_matrix.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(tfidf_matrix)
"""

# method3: word2vec (here use pretrained google2vec model)
"""
google2vec = gensim.models.KeyedVectors.load_word2vec_format('model/GoogleNews-vectors-negative300.bin', binary=True)
processed_features_google2vec = []
for sentence in processed_features:
    context = re.sub(r'^\W+|\W+$', '', sentence)
    context = re.split(r'\W+', context)
    context_embeddings = np.mean([google2vec.wv[w] for w in context if w in google2vec.wv], axis=0)
    if (np.isnan(context_embeddings)).any():
        processed_features_google2vec.append(np.zeros(300))
    else:
        processed_features_google2vec.append(context_embeddings)
"""

# create a new dataset with processed features
processed_df = pd.DataFrame(processed_features_tfidf, columns=[str(i) for i in range(len(processed_features_tfidf[0]))])
processed_df['label'] = labels
processed_df.to_csv('processed_df.csv', index=False)

processed_df = pd.read_csv("processed_df.csv")

# Dividing data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(processed_df.loc[:, processed_df.columns != 'label'],
                                                    processed_df['label'], test_size=0.2, random_state=0)
#X_train, X_test, y_train, y_test = train_test_split(processed_features_google2vec, labels, test_size=0.2, random_state=0)

# Training the model
# method1: RandomForest
text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)

# method2: CNN (not used currently)

# Making predictions and evaluate the model
predictions = text_classifier.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))

con.close()
