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
    #print(word)
    #print(otherword)
    # called distance but is actually similarity
    sim = distance.get_jaro_distance(word, otherword)
    return sim

data = np.array([
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
        'automaty',
        'automatický',
        'antenoly',
        ])

# Reshape the data.
# X = np.arange(len(data)).reshape(-1, 1)
X = data.reshape(-1, 1)
print(X.shape)
print(X)

D = np.array([[]])
for f1 in data:
    for f2 in data:
        D[f1,f2] = jwsim(f1, f2)

#D = pairwise_distances(X, metric=jwsim)

# clustering = AgglomerativeClustering(affinity=jwsim, linkage='average')
clustering = AgglomerativeClustering(affinity='precomputed', linkage='average')

clustering.fit(D)



