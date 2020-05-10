import pandas as pd
import re

# import twitter data related to coronavirus
coronavirus_sentiment = pd.read_csv("corona_processed.csv")

# extracting text data
text_data = coronavirus_sentiment['text'].values

# data cleaning
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

# write the processed text data into txt file
with open("processed_features_corpus.txt", 'w') as output:
    for row in processed_features:
        output.write(str(row) + '\n')