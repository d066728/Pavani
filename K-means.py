import nltk
import numpy as np
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from time import time




f = open("/home/pavani/Desktop/CLEAN.txt", 'r')
contents = []
for line in f.readlines()[1:]:
    contents.append(line)

#remove punc, num, stopwords, strange symbols, transfer to lowercase
contents = ["".join(i for i in t if i not in string.punctuation) for t in contents]
contents = ["".join(i for i in t if i not in string.punctuation) for t in contents]
contents = [re.sub(r'\b\d+\b', '', t) for t in contents]
contents = [re.sub(r'[^a-zA-Z0-9 ]', r'', t) for t in contents]
contents = [t.lower() for t in contents]
contents = [' '.join([word for word in t.split() if word not in stopwords.words("english")]) for t in contents]


    #t = "".join(i for i in t if i not in string.punctuation)
    #t = re.sub(r'\b\d+\b', '', t)
    #t=re.sub(r'[^a-zA-Z0-9 ]', r'', t)
    #t=t.lower()
    #t=' '.join([word for word in t.split() if word not in stopwords.words("english")])
        #print t

vectorizer = TfidfVectorizer(max_df=0.5, max_features=1000,min_df=3,use_idf=True,smooth_idf=True,ngram_range=(1,2),analyzer='word',token_pattern='\w{5,}')
X = vectorizer.fit_transform(contents)

#km = KMeans(n_clusters=4, init='k-means++', max_iter=100, n_init=1,
#            verbose=True)
km = MiniBatchKMeans(n_clusters=10, init='k-means++', n_init=1,
                     init_size=1000, batch_size=1000, verbose=True)

t0 = time()
km.fit(X)
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(20):
    print ("Cluster %d:" % i)
    for ind in order_centroids[i, :20]:
        print (' %s' % terms[ind])
print("Clustering sparse data with %s" % km)
print("done in %0.3fs" % (time() - t0))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))





