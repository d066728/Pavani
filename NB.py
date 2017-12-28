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
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import metrics
from array import array
from sklearn import preprocessing



f = open("/home/pavani/Desktop/CLEAN.txt",'r')
contents = []
for line in f.readlines()[1:]:
    contents.append(line.split('|')[1] + line.split('|')[2])
#print type(contents[1])
#contents = [re.sub(r'[^a-zA-Z ]', r'', t) for t in contents]


f = open("/home/pavani/Desktop/CLEAN.txt",'r')
tags = []
for line in f.readlines()[1:]:
    tagsStr = line.split('|')[3].rstrip()
    tagsInRow = tagsStr.split('~')
    tagsInRow = list(filter(None, tagsInRow))
    tags.append(tagsInRow)

#print type(tags[1])
#tags = [re.sub(r'[^a-zA-Z ]', r'', t) for t in contents]
#print('tags:' + str(tags))


#remove punc, num, stopwords, strange symbols, transfer to lowercase
contents = ["".join(i for i in t if i not in string.punctuation) for t in contents]
contents = ["".join(i for i in t if i not in string.punctuation) for t in contents]
contents = [re.sub(r'\b\d+\b', '', t) for t in contents]
contents = [re.sub(r'[^a-zA-Z0-9 ]', r'', t) for t in contents]
contents = [t.lower() for t in contents]
contents = [' '.join([word for word in t.split() if word not in stopwords.words("english")]) for t in contents]



vectorizer = TfidfVectorizer(max_df=0.5, max_features=4450,min_df=3,use_idf=True,smooth_idf=True,ngram_range=(1,2),lowercase=True,analyzer='word',token_pattern='\w{5,}')
terms = vectorizer.fit_transform(contents)


X_test = vectorizer.transform(contents)
features_by_gram = defaultdict(list)

for f, w in zip(vectorizer.get_feature_names(), vectorizer.idf_):
    features_by_gram[len(f.split(' '))].append((f, w))
top_n = 4450


t0 = time()
#terms = vectorizer.get_feature_names()
for gram, features in features_by_gram.iteritems():
    top_features = sorted(features, key=lambda x: x[1], reverse=True)[:top_n]
    top_features = [f[0] for f in top_features]
    print '{}-gram top:'.format(gram), top_features
    
#terms = np.array(terms).reshape(-1, 1)
#tags  = np.array(tags).reshape(-1, 1)
tags  = np.array(tags)
print(terms)
print(tags)
#arr = np.array(array)
# terms== arr.float64
# tags== arr.float6


from sklearn.preprocessing import MultiLabelBinarizer
tags = MultiLabelBinarizer().fit_transform(tags)
print(tags)

termsTraining = terms[:3000,:]
termsTest = terms[3000:,:]

tagsTraining = tags[:3000,:]
tagsTest = tags[3000:,:]

from skmultilearn.problem_transform import LabelPowerset

# instantiate a Multinomial Naive Bayes model
nb = LabelPowerset(GaussianNB())
nb.fit(termsTraining.toarray(), tagsTraining)

 
 

#  make class predictions for X_test_dtm
y_pred = nb.predict(termsTest)
score = metrics.accuracy_score(tagsTest, y_pred)
print('score', score)
#matrix = metrics.confusion_matrix(tags, y_pred_class)
#print('matrix', matrix)

#contents_test[y_pred_class > tags]
#y_pred_prob = nb.predict_proba(termsTest)[:, 1]

# calculate AUC
#metrics.roc_auc_score(tagsTest, y_pred_prob)
#print (metrics.roc_auc_score(tags, y_pred_prob))



















