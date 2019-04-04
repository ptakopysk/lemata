#!/usr/bin/env python3
#coding: utf-8

import argparse
import sys
from collections import defaultdict

import numpy as np

from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics.pairwise import pairwise_distances

from scipy.sparse import dok_matrix

from pyjarowinkler import distance

def jwdist(word, otherword):
    # called distance but is actually similarity
    return 1-distance.get_jaro_distance(word, otherword)

index2word = [
        'auto',
        'automat',
        'autogram',
        'autobus',
        'autograf',
        'anténa',
        'anihilace',
        'antenola',
        'asimilace',
        'astenický',
        'autoerotický',
        'erotický',
        'excentrický',
        'automaty',
        'automatický',
        'antenoly',
        ]

N = len(index2word)

def get_stem(form):
    #return form[:2]
    return form[:1]
    #return 'a'

word2index = dict()
index = 0
for index, word in enumerate(index2word):
    word2index[word] = index

# D = np.zeros((N,N))
# sparse matrix
D = dok_matrix((N,N), dtype=float)
# TODO only iterate over what makes sense
for i1 in range(N):
    for i2 in range(N):
        if get_stem(index2word[i1]) == get_stem(index2word[i2]):
            D[i1,i2] = jwdist(index2word[i1], index2word[i2])

#print(D)

DD = np.zeros((N,N))
for i1 in range(N):
    for i2 in range(N):
        DD[i1,i2] = jwdist(index2word[i1], index2word[i2])

print(DD)

L = 'average'
#L = 'single'
#L = 'complete'
clustering = AgglomerativeClustering(affinity='precomputed',
        linkage=L,
        connectivity = D,
        compute_full_tree = False,
        n_clusters=3)

labels = clustering.fit_predict(DD)
# print(labels)

label2words = defaultdict(list)
for index in range(N):
    label2words[labels[index]].append(index)

for label in label2words:
    print('Cluster', label)
    for index in label2words[label]:
        print(index2word[index])
    print()


