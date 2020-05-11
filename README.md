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

```
```




