#encoding=utf-8
import os
import re
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

import nltk.data
#nltk.download()
#from nltk.corpus import stopwords

from gensim.models.word2vec import Word2Vec
def load_dataset(name, nrows=None):
    datasets = {
        'unlabeled_train': 'unlabeledTrainData.tsv',
        'labeled_train': 'labeledTrainData.tsv',
        'test': 'testData.tsv'
    }
    if name not in datasets:
        raise ValueError(name)
    data_file = os.path.join('..', 'data', datasets[name])
    df = pd.read_csv(data_file, sep='\t', escapechar='\\', nrows=nrows)
    print('Number of reviews: {}'.format(len(df)))
    return df
df = load_dataset('unlabeled_train')
df.head()
def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)#去掉标点等
    text = text.lower().split()#转小写,分词
    return text;
sentence=df.review.apply(clean_text)
sentences=sum(sentence,[])
# 设定词向量训练的参数
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

model_name = '{}features_{}minwords_{}context.model'.format(num_features, min_word_count, context)
print('Training model...')
model = Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model.save(os.path.join('..', 'models', model_name))
model.most_similar("man")
print model.most_similar("man")