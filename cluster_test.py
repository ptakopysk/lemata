#!/usr/bin/env python3
#coding: utf-8

import argparse
import sys
from collections import defaultdict

import numpy as np

from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics.pairwise import pairwise_distances

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

word2index = dict()
index = 0
for index, word in enumerate(index2word):
    word2index[word] = index


D = np.zeros((N,N))
for i1 in range(N):
    for i2 in range(N):
        D[i1,i2] = jwdist(index2word[i1], index2word[i2])

clustering = AgglomerativeClustering(affinity='precomputed', linkage='average')

labels = clustering.fit_predict(D)
# print(labels)

label2words = defaultdict(list)
for index in range(N):
    label2words[labels[index]].append(index)

for label in label2words:
    print('Cluster', label)
    for index in label2words[label]:
        print(index2word[index])
    print()


