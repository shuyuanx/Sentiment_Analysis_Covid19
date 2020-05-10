import pandas as pd
import numpy as np
from random import randrange
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import random
import time
import multiprocessing as mp
#from multiprocessing import freeze_support
from itertools import repeat


'''reference: https://machinelearningmastery.com/implement-random-forest-scratch-python/'''

start_time = time.time()

# number of processes
P = 4
# A thread pool of P processes
pool = mp.Pool(P)


# readin the csv file
processed_df = pd.read_csv("processed_df.csv")

# sample the observations
#sampled_df = processed_df.sample(frac=0.5, replace=False, random_state=0)
sampled_df = processed_df.sample(frac=0.2, replace=False, random_state=0)

# dividing data into training and test sets
#train, test = train_test_split(processed_df, test_size=0.2, random_state=0)
train, test = train_test_split(sampled_df, test_size=0.2, random_state=0)


''' writing random forest classifier from scratch'''


 
# Convert class column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = {}
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# calculate accuracy rate 
def accuracy(actual, predicted):
    correct = sum(np.array(actual) == np.array(predicted))
    acc = correct/float(len(actual))
    return acc



# split dataset based on an attribute and a value of the attribute
def test_split(idx, val, dataset):
    left, right = [], []
    for row in dataset:
        if row[idx] < val:
            left.append(row)
        else:
            right.append(row)
    return left, right

# calculate gini index for split dataset
def gini_index(groups, classes):
    # count all samples at the split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted gini index for each group
    gini_sum = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val)/float(size)
            score += p**2
        # weight the group score by its relative size
        gini_sum += (1.0-score)*(float(size)/n_instances)
    return gini_sum

# select the best split point for a dataset
def get_split(dataset, n_features):
    classes = list(set(row[-1] for row in dataset))
    b_idx, b_val, b_score, b_groups = 999, 999, 999, None
    features = random.sample(range(len(dataset[0])), n_features)
    for idx in features:
        for row in dataset:
            groups = test_split(idx, row[idx], dataset)
            gini = gini_index(groups, classes)
            if gini < b_score:
                b_idx, b_val, b_score, b_groups = idx, row[idx], gini, groups
    return {'index': b_idx, 'value': b_val, 'groups': b_groups}

# create a terminal node value
def to_terminal(group):
    results = [row[-1] for row in group]
    return max(set(results), key=results.count)

# create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left+right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
	    node['left'] = get_split(left, n_features)
	    split(node['left'], max_depth, min_size, n_features, depth+1)
    # process right child
    if len(right) <= min_size:
	    node['right'] = to_terminal(right)
    else:
	    node['right'] = get_split(right, n_features)
	    split(node['right'], max_depth, min_size, n_features, depth+1)

# build a decision tree
def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root

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

# create a random subsample from the dataset
def subsample(dataset, ratio):
    sample = []
    n_sample = round(len(dataset)*ratio)
    while len(sample) < n_sample:
        idx = randrange(len(dataset))
        sample.append(dataset[idx])
    return sample

# make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = pool.starmap(predict, zip(trees,repeat(row)))
    #predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

# random forest algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    samples = pool.starmap(subsample, zip(repeat(train),[sample_size for i in range(n_trees)]))
    trees = pool.starmap(build_tree, zip(samples,repeat(max_depth),repeat(min_size),repeat(n_features)))
    predictions = [bagging_predict(trees, row) for row in test]
    return predictions


# test the random forest algorithm
train_data, test_data = np.array(train), np.array(test)

# convert class column to integers
str_column_to_int(train_data, len(train_data[0])-1)
str_column_to_int(test_data, len(test_data[0])-1)


n_features = int(sqrt(len(train_data[0])-1))
# predicted = random_forest(train_data, test_data, 30, 1, 1.0, 200, n_features=n_features)
predicted = random_forest(train_data, test_data, 30, 1, 0.2, 10, n_features=n_features)

y_test = test_data[:,-1]
acc = accuracy(y_test, predicted)
print(acc)
print(confusion_matrix(list(y_test),predicted))
print(classification_report(list(y_test),predicted))

print("--- %s seconds ---" % (time.time() - start_time))


