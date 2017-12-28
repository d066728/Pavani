import nltk
import itertools
from operator import itemgetter
from pygraph.classes.graph import graph
from pygraph.classes.digraph import digraph
from pygraph.algorithms.pagerank import pagerank
from pygraph.classes.exceptions import AdditionError
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
from sklearn import preprocessing
from sklearn import metrics
import io
import itertools
import networkx as nx
import nltk
import os
from pattern.en import singularize
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

f = open("/home/pavani/Desktop/content.txt",'r')
contents = []
for line in f.readlines()[1:]:
    contents.append(line.split('|')[1])


#remove punc, num, stopwords, strange symbols, transfer to lowercase
contents = ["".join(i for i in t if i not in string.punctuation) for t in contents]
contents = ["".join(i for i in t if i not in string.punctuation) for t in contents]
contents = [re.sub(r'\b\d+\b', '', t) for t in contents]
contents = [re.sub(r'[^a-zA-Z0-9 ]', r'', t) for t in contents]
contents = [t.lower() for t in contents]
contents = [' '.join([word for word in t.split() if word not in stopwords.words("english")]) for t in contents]
#contents = [re.sub('(\S.+?[.!?])(?=\s+|$)|(\S.+?)(?=[\n]|$)','',t) for t in contents]
contents = [re.sub("([A-Z][a-z]{1,2}\.)\s(\w)",'',t) for t in contents]
contents = [re.sub("(\.[a-zA-Z]\.)\s(\w)",'',t) for t in contents]
contents = [re.sub("([a-zA-Z])\.([a-zA-Z])\.",'',t) for t in contents]
SEPARATOR = r"@"
contents = [re.sub("([A-Z][a-z]{1,2}\.)" + SEPARATOR + "(\w)",'',t) for t in contents]
contents = [re.sub("(\.[a-zA-Z]\.)" + SEPARATOR + "(\w)",'',t) for t in contents]
#print(contents)

from summa import keywords
for text in contents:
    print(text)
    text = nltk.word_tokenize(text)

    tagged = nltk.pos_tag(text)


def filter_for_tags(tagged, tags=['NN', 'JJ', 'NNP']):
    return [item for item in tagged if item[1] in tags]


def normalize(tagged):
    return [(item[0].replace('.', ''), item[1]) for item in tagged]


def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in itertools.ifilterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element

tagged = filter_for_tags(tagged)
tagged = normalize(tagged)
print()

unique_word_set = unique_everseen([x[0] for x in tagged])


gr = digraph()
gr.add_nodes(list(unique_word_set))


window_start = 0
window_end = 2

while 1:

    window_words = tagged[window_start:window_end]
    if len(window_words) == 2:
        print window_words
        try:
            gr.add_edge((window_words[0][0], window_words[1][0]))
        except AdditionError, e:
            print 'already added %s, %s' % ((window_words[0][0], window_words[1][0]))
    else:
        break

    window_start += 1
    window_end += 1

calculated_page_rank = pagerank(gr)
di = sorted(calculated_page_rank.iteritems(), key=itemgetter(1))
for k, g in itertools.groupby(di, key=itemgetter(1)):
    print k, map(itemgetter(0), g)
