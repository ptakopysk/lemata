#!/usr/bin/env python3
#coding: utf-8

import argparse
import sys
from collections import defaultdict
from sortedcollections import ValueSortedDict
from collections import OrderedDict

from sklearn.metrics import accuracy_score
from numpy import inner
from numpy.linalg import norm

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from ast import literal_eval as make_tuple
from sklearn.metrics import confusion_matrix
import itertools

from sklearn.cluster import AgglomerativeClustering

from pyjarowinkler import distance

import unidecode

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(
        description='find lemma for form as nearest lemma in emb space')
ap.add_argument('embeddings',
        help='file with the embeddings')
ap.add_argument('conllu_all',
        help='file with the forms and lemmas')
ap.add_argument('conllu_test',
        help='file with the forms and lemmas')
ap.add_argument("-n", "--number", type=int,
        help="How many embeddings to read in")
ap.add_argument("-V", "--verbose", action="store_true",
        help="Print more verbose progress info")
ap.add_argument("-N", "--normalize", action="store_true",
        help="Normalize the embeddings")
ap.add_argument("-p", "--plot", type=str,
        help="Plot the dendrogramme for the given stem")
ap.add_argument("-s", "--similarity", type=str,
        help="Similarity: cos or jw")
ap.add_argument("-S", "--stems", action="store_true",
        help="only look for words with the same stem")
ap.add_argument("-L", "--length", type=float, default=0.05,
        help="Weight for length similarity")
ap.add_argument("-l", "--lowercase", action="store_true",
        help="lowercase input forms")
ap.add_argument("-C", "--cut", type=int, default=100,
        help="Cut down the number of most similar words to 100 for each word")
ap.add_argument("-U", "--upperbound", action="store_true",
        help="Highest achievable accuracy given the settings (esp. stem pruning)")
args = ap.parse_args()






# https://github.com/scikit-learn/scikit-learn/blob/70cf4a676caa2d2dad2e3f6e4478d64bcb0506f7/examples/cluster/plot_hierarchical_clustering_dendrogram.py
# Authors: Mathew Kallada
# License: BSD 3 clause
"""
=========================================
Plot Hierarachical Clustering Dendrogram 
=========================================
This example plots the corresponding dendrogram of a hierarchical clustering
using AgglomerativeClustering and the dendrogram method available in scipy.
"""

import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

def plot_dendrogram(model, **kwargs):


    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

    plt.xticks(rotation=90)
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.2)








def embsim(word, otherword):
    emb1 = embedding[word]
    emb2 = embedding[otherword]
    return inner(emb1, emb2)/(norm(emb1)*norm(emb2))

def jwsim(word, otherword):
    # called distance but is actually similarity
    sim = distance.get_jaro_distance(word, otherword)
    uword = unidecode.unidecode(word)
    uotherword = unidecode.unidecode(otherword)
    usim = distance.get_jaro_distance(uword, uotherword)    
    return (sim+usim)/2

def lensim(word, otherword):
    return 1 / (1 + args.length * abs(len(word) - len(otherword)) )

def similarity(word, otherword):
    if args.similarity == 'jw':
        return jwsim(word, otherword)
    elif args.similarity == 'jwxcos':
        return jwsim(word, otherword) * embsim(word, otherword)
    elif args.similarity == 'jwxcosxlen':
        return jwsim(word, otherword) * embsim(word, otherword) * lensim(word, otherword);
    elif args.similarity == 'len':
        return lensim(word, otherword);
    else:
        # cos
        return embsim(word, otherword)

print('Read in embeddings', file=sys.stderr)
embedding = defaultdict(list)
with open(args.embeddings) as embfile:
    size, dim = map(int, embfile.readline().split())
    if args.number:
        size = min(size, args.number)
    for i in range(size):
        fields = embfile.readline().split()
        embedding[fields[0]] = list(map(float, fields[1:]))
        if args.normalize:
            embedding[fields[0]] /= norm(embedding[fields[0]])

def get_stem(form):
    return form[:2]

print('Read in forms and lemmas', file=sys.stderr)
forms = set()
lemmas = set()  # not currently used
forms_stemmed = defaultdict(set)
with open(args.conllu_all) as conllufile:
    for line in conllufile:
        fields = line.split()
        if fields and fields[0].isdecimal():
            assert len(fields) > 2
            form = fields[1]
            lemma = fields[2]
            # pos = fields[3]
            if args.lowercase:
                form = form.lower()
                lemma = lemma.lower()
            if form in embedding:
                forms.add(form)
                forms_stemmed[get_stem(form)].add(form)
            if lemma in embedding:
                lemmas.add(lemma)
                #forms.add(lemma)
                #forms_stemmed[get_stem(lemma)].add(lemma)

print('Read in test form-lemma pairs', file=sys.stderr)
test_data = list()
with open(args.conllu_test) as conllufile:
    for line in conllufile:
        fields = line.split()
        if fields and fields[0].isdecimal():
            assert len(fields) > 2
            form = fields[1]
            lemma = fields[2]
            # pos = fields[3]
            if args.lowercase:
                form = form.lower()
                lemma = lemma.lower()
            test_data.append((form, lemma))
print('Done reading', file=sys.stderr)

def get_dist(form1, form2):
    # similarity to distance
    # if form1 != form2 and form1 in embedding and form2 in embedding:
    if form1 in embedding and form2 in embedding:
        return -similarity(form1, form2)
    else:
        return None


def get_sim(form1, form2):
    if form1 in dists and form2 in dists[form1]:
        # distance to similarity
        return -dists[form1][form2]
    else:
        return -2


# cluster each hypercluster

L = 'average'
iterate_over = forms_stemmed.items()
if args.plot:
    iterate_over = [args.plot]

for stem in iterate_over:
    # vocabulary
    forms = forms_stemmed[stem]
    index2word = list(forms)
    word2index = dict()
    for index, word in enumerate(index2word):
        word2index[word] = index

    I = len(index2word)
    
    D = np.empty((I, I))
    for i1 in range(I):
        for i2 in range(I):
            D[i1,i2] = get_dist(index2word[i1], index2word[i2])

    C = max(int(I/10), 2)

    clustering = AgglomerativeClustering(affinity='precomputed',
            linkage = L,
            compute_full_tree = True,
            n_clusters=C)

    if args.verbose:
        print(stem)
        print(forms)
        print(I)
        print(C)
        print(dir(clustering))
        #print(D)

    if (I > 1):
        labels = clustering.fit_predict(D)
        print(dir(clustering))
    else:
        # just one word -> just one cluster
        assert I == 1
        C = 1
        labels = [0]

    label2words = defaultdict(list)
    for i in range(I):
        label2words[labels[i]].append(i)

    print('Stem', stem)
    for label in range(C):
        print('Cluster', label)
        for index in label2words[label]:
            print(index2word[index])
        print()

    if args.plot:
        # at the i-th iteration, children[i][0] and children[i][1] are merged to form node n_samples + i
        index = 0
        for merge in clustering.children_:
            print(index, merge[0], merge[1])
            index += 1

        plt.title('Hierarchical Clustering Dendrogram')
        plot_dendrogram(clustering, labels=index2word)
        plt.show()



