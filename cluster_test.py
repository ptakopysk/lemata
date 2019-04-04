#!/usr/bin/env python3
#coding: utf-8

import argparse
import sys
from collections import defaultdict

import numpy as np

from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics.pairwise import pairwise_distances

from pyjarowinkler import distance

def jwsim(word, otherword):
    # called distance but is actually similarity
    sim = distance.get_jaro_distance(word, otherword)
    return sim

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
        D[i1,i2] = jwsim(index2word[i1], index2word[i2])

print(D)

#D = pairwise_distances(X, metric=jwsim)

# clustering = AgglomerativeClustering(affinity=jwsim, linkage='average')
clustering = AgglomerativeClustering(affinity='precomputed', linkage='average')

clustering.fit(D)



