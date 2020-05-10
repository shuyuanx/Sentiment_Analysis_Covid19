#!/usr/bin/python

import sys
import csv

previous = None
sum = 0
count = 0

doc_freq_dict = {}
for line in sys.stdin:
    key, value = line.split( '\t' )
    
    if key != previous:
        if previous is not None:
            # When building the vocabulary ignore terms that have a document frequency (absolute counts)
            # strictly lower the given threshold 7 (corpus-specific stop words).
            if sum > 7:
                doc_freq_dict.update({previous: sum})
        previous = key
        sum = 0
    
    sum = sum + int( value )
    count = count + 1

# When building the vocabulary ignore terms that have a document frequency
# strictly higher than the given threshold (corpus-specific stop words).
if sum <= 7:
    doc_freq_dict.update({previous: sum})

for key in doc_freq_dict.keys():
    # When building the vocabulary ignore terms that have a document frequency (absolute counts)
    # strictly higher the given threshold 0.8 (corpus-specific stop words).
    if doc_freq_dict.get(key) > count * 0.8:
        doc_freq_dict.pop(key)

# save the document frequency dictionary as a csv file
with open('doc_freq.csv', 'w') as f:
    w = csv.writer(f)
    w.writerows(doc_freq_dict.items())
