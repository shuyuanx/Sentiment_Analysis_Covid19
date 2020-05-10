import numpy as np 
import pandas as pd 

df = pd.read_csv("tfidf_matrix.csv", header=None)
labels = pd.read_fwf('labels.txt', header=None)

df['label'] = labels.iloc[:,0]
processed_df.to_csv('processed_df.csv', index=False)
