from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import nltk
import numpy as np
import re
import string
import json
import gensim
import math
import logging
from nltk.corpus import stopwords
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from time import time
from collections import defaultdict
from array import array
from sklearn import preprocessing
from gensim import matutils, models
from gensim.matutils import Sparse2Corpus, corpus2csc
from gensim.models import LdaModel
from gensim.models import VocabTransform
from sklearn.linear_model import LogisticRegression
 


f = open("/home/pavani/Desktop/CLEAN.txt",'r')
contents = []
for line in f.readlines()[1:]:
    contents.append(line.split('|')[1] + line.split('|')[2])



#remove punc, num, stopwords, strange symbols, transfer to lowercase
contents = ["".join(i for i in t if i not in string.punctuation) for t in contents]
contents = ["".join(i for i in t if i not in string.punctuation) for t in contents]
contents = [re.sub(r'\b\d+\b', '', t) for t in contents]
contents = [re.sub(r'[^a-zA-Z0-9 ]', r'', t) for t in contents]
contents = [t.lower() for t in contents]
contents = [' '.join([word for word in t.split() if word not in stopwords.words("english")]) for t in contents]

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])


no_features = 1000

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(contents)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20


# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

no_top_words = 10
display_topics(lda, tf_feature_names, no_top_words)
