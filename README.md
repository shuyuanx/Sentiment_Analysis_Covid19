# Sentiment_Analysis_Covid19
Repository for CS205 Final Project, Spring 2020

# CS 205 Final Project

Names: Weiru Chen, Shuyuan Xiao, Wanxi Yang, Zhao Lyu

## Problem Statement

Coronaviruses are a group of viruses that could cause illness in humans and some animals. The COVID-19 disease is defined as an illness that is caused by severe acute respiratory syndrome coronavirus 2. On March 11th, 2020, WHO declared the coronavirus a global pandemic, signaling its first such designation since the declaration of H1N1 influenza as a pandemic in 2009. There are more than 1 million confirmed cases in the U.S, and more than 3 million confirmed cases around the world, and those numbers are rapidly growing.

How do people cope with this pandemic? What is the general attitude towards the coronavirus? To understand how people describe the coronavirus, we want to analyze textual data that are related to coronavirus and observe the trend of emotions and the topics people talk about in general.

## Solution & Existing Work

Our project aims to use data on Twitter to analyze people’s attitudes towards coronavirus. Twitter is a social media full of people’s textual description of their emotions and the events happened in life. We obtained a dataset on coronavirus and performed sentiment classification on them. In our project, we mainly focus on Naive Bayes and Random Forest models and use multithreading methods to accelerate the analyzing process.

  

Twitter sentiment analysis is a common form of data analysis as Twitter is one of the most easily-obtained data sources. There exist many works on different ways of analyzing tweets with a diverse selection of models. These previous projects often focus on experimenting with new algorithms for performing sentiment analysis. Our work is distinct from the existing work in the following ways:

1. We focus specifically on tweets that involve Covid-19

2. Our dataset is relatively large compared to other mini-dataset used for sentiment analysis

2. We pre-process the data using Spark.

3. We take advantage of multiprocessing and implement that into our Naive Bayes and Random Forest models.


Our project requires multi-processing because

1. Big data: we have a relatively large datasets

2. Big compute: the models we use can benefit from multi-processing

3. Using multi-processing for natural language processing is fairly uncommon, we want to use this opportunity to explore this area.

## Data

The main datasets we used in the project are collected from Twitter. Because the Twitter developer’s account’s application process takes much longer during the pandemic, we ultimately used a tool called GetOldTweets3 to scrape the Twitter information. The csv contains 11 columns: id (str), permalink (str), username (str), to (str), text (str), data (datetime) in UTC, retweets (int), favorites (int), mentions (str) hashtags (str), and geo (str). The queries are constricted to the English language using the option --lang en.

  

There are two main datasets. The first data set consists of about 1,6000 tweets that include “coronavirus” on May 2nd, 2020. The second data set consists of about 10,000 tweets that include “coronavirus” from May 2nd to May 8th 2020. Because Twitter tends to limit API’s visits, these are not the complete tweets that include “coronavirus.”
 

Because we did not tag the sentiment of the tweets we collected, we used two other Twitter datasets as our training material. For the Naive Bayes model, we used the Sentiment140 dataset on Kaggle for training. For the Random Forest model, we used the Twitter Airline Sentiment dataset on Kaggle for training.

<img src="./images/data_processing.png">


## Technical Description
- Random Forest Model:
  - Programming language: Python 3.7.4
  - Python libraries: 
    - Numpy, Pandas, re, nltk, sqlite3, sklearn, gensim, collections, sys 
    - csv, multiprocessing, threading, random, time, itertools, textblob 
  - PySpark libraries: 
    - pyspark.sql, pyspark.ml, pyspark.mllib 
  - PySpark environment (see performance evaluation for more information): 
    - AWS EC2 environment 
    - Ubuntu 16.04 
    - m4.xlarge instance 
 
## Model
### Naive Bayes Model:
Previously we have seen how the Bayes model will help us in sentiment analysis using prior and posterior. Before calculating any probabilities for posteriors, we need to first preprocess our data as they are not numerical data type. Overall we are only interested in the sentiment, users’ attitudes, and text, users’ tweets. Therefore, we group sentiments into three categories: positive, neutral and negative. Moreover, we apply the bag of words (BOW) to text. That is, we map each text into a list of strings, and therefore we can calculate priors and posteriors based on appearances of strings. This will definitely be a big dataset that grows even more with time. Further, timely feedback is necessary in dealing with sentimental problems since we are interested in how users’ attitudes change. So the task would fail if feedback can not be generated in a timely manner, possibly due to slow process in processing and digesting the huge dataset. Note that ordering is not required in BOW, so we could optimize priors and posteriors calculations in Bayes model.

To run Bayes, we need to have priors set up by training data, where we record the probability of each word appearing. Then in the prediction method, we will use these priors for posteriors. (TODO: May add formula to show how Bayes works) Again, no ordering is needed here, so we could apply multiprocessing, specifically pool method, to update posteriors of positive, neutral, and negative text by testing data.

In our code, we have applied Bayes to a sample of size 100,000 out of 1.6 million tweets, 80% for training and 20% for testing. We achieved 83% accuracy overall in the Bayes model, and we will apply this trained Bayes model to predict fresh tweets lately to see users’ attitudes with time varying. This gives the number of positive and negative tweets given different dates in May.

<img src="./images/sentiment.png">
<img src="./images/table.png">

We can see that the number of negative tweets are all greater than that of positive tweets during these days. It is not surprising as we see many concerns coming with COVID19 these days. Moreover, it is worth noticing that the number of total tweets overall is increasing over time, so simply checking the difference between the number of positive and negative tweets would be somehow biased. It would be interesting to ask how much these concerns and negative tweets result from COVID19, how positive tweets connect with development of COVID19 updates, and other questions developing over time specifically convolved with COVID19 timely updates.

### Random Forest Model:

Before the implementation of the random forest model, we need to preprocess the text data of tweets and convert them into numerical data. Here we use TF-IDF (Term Frequency Inverse Document Frequency) method to convert the text data into numerical vectors. This process can be parallelized by using MapReduce and Multiprocessing. And the performance of these parallelization methods will be discussed in later sections.

We used the random forest model to train and classify the twitter-airline-sentiment tweets into positive, neutral, and negative. Then we used this model to predict the sentiments of coronavirus related tweets. The overall accuracy score of the random forest model is around 0.9379. The confusion matrix of the results is shown as follows:

<img src="./images/confusion_matrix_rfc_model.png">

We can see that the performance of this random forest model is relatively high and the imbalanced data does not lead to highly biased predictions. We have also used the random forest model from a library in PySpark, but the accuracy rate is not as high as this model. So we use this model for future sentiment predictions.

While the random forest model is implemented by hand and the dataset is quite large, it takes extremely long time to train the algorithm and make predictions and evaluations. So parallel computing can be applied in this process in order to reduce the program run time. Here we mainly use the Multiprocessing module in Python for parallelization, and Multithreading is also used in some functions to further reduce the run time. The performance of these two parallelization methods will also be discussed in later sections.

After the evaluation of this random forest model, we save this model and use it to predict the sentiments of the 16900 coronavirus related tweets scraped from Twitter. The predicted results is shown in the following plot:

<img src="./images/prediction_result.png">

We can see from the plot that the number of tweets that are predicted as negative is the highest, while the number of tweets predicted as positive is the lowest. 

We also predicted the sentiments of another dataset with 114496 coronavirus related tweets scraped from Twitter. The predicted results is shown in the following plot:

<img src="./images/prediction_over_time.png">

From this plot we can see that for this larger dataset, the number of tweets that are predicted as negative is slightly lower than the number of tweets predicted as neutral and is much higher than the number of tweets predicted as positive over time (from May 2nd to May 8th). 



