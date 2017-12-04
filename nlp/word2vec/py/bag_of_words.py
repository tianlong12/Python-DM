# -*- coding:utf-8 -*-
import os
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import nltk
#nltk.download()
from nltk.corpus import stopwords

def display(text, title):
    print(title)
    print("\n----------我是分割线-------------\n")
    print(text)

datafile = os.path.join('..', 'data', 'labeledTrainData.tsv')
df = pd.read_csv(datafile, sep='\t', escapechar='\\')
print('Number of reviews: {}'.format(len(df)))
print df.head()
raw_example = df['review'][1]
display(raw_example, '原始数据')
example = BeautifulSoup(raw_example, 'html.parser').get_text()
display(example, '去掉HTML标签的数据')
example_letters = re.sub(r'[^a-zA-Z]', ' ', example)
display(example_letters, '去掉标点的数据')
words = example_letters.lower().split()
display(words, '纯词列表数据')
stopwords = {}.fromkeys([ line.rstrip() for line in open('../stopwords.txt')])
words_nostop = [w for w in words if w not in stopwords]
display(words_nostop, '去掉停用词数据')
eng_stopwords = set(stopwords)
#数据预处理
def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)#去掉标点等
    words = text.lower().split()#转小写，划分为词
    words = [w for w in words if w not in eng_stopwords]#去掉停用词
    return ' '.join(words)
clean_text(raw_example)
df['clean_review'] = df.review.apply(clean_text)
df.head()
vectorizer = CountVectorizer(max_features = 5000)#取单词频次最高的5000个词作为词典
train_data_features = vectorizer.fit_transform(df.clean_review).toarray()#可以输出看一下，共有5000列，如果有词典当中的单词，则为1，否则为0
forest = RandomForestClassifier(n_estimators = 10)
forest = forest.fit(train_data_features, df.sentiment)
confusion_matrix(df.sentiment, forest.predict(train_data_features))
del df
del train_data_features
datafile = os.path.join('..', 'data', 'testData.tsv')
df = pd.read_csv(datafile, sep='\t', escapechar='\\')
print('Number of reviews: {}'.format(len(df)))
df['clean_review'] = df.review.apply(clean_text)
df.head()
test_data_features = vectorizer.transform(df.clean_review).toarray()
result = forest.predict(test_data_features)
output = pd.DataFrame({'id':df.id,'sentiment':result})
output.head()
output.to_csv(os.path.join('..', 'data', 'Bag_of_Words_model.csv'), index=False)
del df
del test_data_features
