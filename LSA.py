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
from sklearn.decomposition import TruncatedSVD


filename= '/home/pavani/Desktop/comapre.txt'

f = open( filename, "r" )
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

vectorizer = TfidfVectorizer(max_df=0.5, max_features=1000,min_df=3,use_idf=True,smooth_idf=True,ngram_range=(1,3),analyzer='word',token_pattern='\w{5,}')
X = vectorizer.fit_transform(contents)

print(X)
X.shape
lsa = TruncatedSVD(n_components=27, n_iter=100)
lsa.fit(X)
lsa.components_[0]
import sys
print (sys.version)

terms = vectorizer.get_feature_names()
for i, comp in enumerate(lsa.components_): 
    termsInComp = zip (terms,comp)
    sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=True) [:10]
    print("Concept %d:" % i )
    for term in sortedTerms:
        print(term[0])
    print (" ")
