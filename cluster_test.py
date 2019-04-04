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

hyperclusters = defaultdict(set)
for index, word in enumerate(index2word):
    hyperclusters[get_stem(word)].add(index)

L = 'average'
#L = 'single'
#L = 'complete'
C = 2
for stem in hyperclusters:
    indices = hyperclusters[stem]
    I = len(indices)

    # index -- global, all words
    # local index -- local for indices, 0-based sequence
    # or maybe: do not use the global indices at all, just compute the
    # index2word and word2index locally
    index2li = dict()
    li2index = dict()
    for li, index in enumerate(indices):
        index2li[index] = li
        li2index[li] = index

    D = np.empty((I, I))
    for i1, li1 in index2li.items():
        for i2, li2 in index2li.items():
            D[li1,li2] = jwdist(index2word[i1], index2word[i2])

    clustering = AgglomerativeClustering(affinity='precomputed',
            linkage = L,
            compute_full_tree = False,
            n_clusters=C)

    labels = clustering.fit_predict(D)

    label2words = defaultdict(list)
    for i, li in index2li.items():
        label2words[labels[li]].append(i)

    print('Stem', stem)
    for label in range(C):
        print('Cluster', label)
        for index in label2words[label]:
            print(index2word[index])
        print()


