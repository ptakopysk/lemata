#!/usr/bin/env python3
#coding: utf-8

# if avg linkage > 0.30 stop

import argparse
import sys
from collections import defaultdict
from sortedcollections import ValueSortedDict
from collections import OrderedDict

from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import homogeneity_completeness_v_measure

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

import logging

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
ap.add_argument("-t", "--threshold", type=float, default=0.30,
        help="Do not perform merges with avg distance greater than this")
ap.add_argument("-p", "--plot", type=str,
        help="Plot the dendrogramme for the given stem")
ap.add_argument("-m", "--merges", action="store_true",
        help="Write out the merges")
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



level = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=level)




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






# TODO WTF ?!

def embsim(word, otherword):
    emb1 = embedding[word]
    emb2 = embedding[otherword]
    sim = inner(emb1, emb2)/(norm(emb1)*norm(emb2))
    # sim = cosine_similarity([emb1], [emb2])
    # assert sim >= -1 and sim <= 1, "Cos sim must be between -1 and 1"
    # shift to 0..1 range
    sim = (sim+1)/2
    return sim

def jwsim(word, otherword):
    # called distance but is actually similarity
    sim = distance.get_jaro_distance(word, otherword)
    uword = unidecode.unidecode(word)
    uotherword = unidecode.unidecode(otherword)
    usim = distance.get_jaro_distance(uword, uotherword)    
    sim = (sim+usim)/2
    assert sim >= 0 and sim <= 1, "JW sim must be between 0 and 1"
    return sim

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

logging.info('Read in embeddings')
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

logging.info('Read in forms and lemmas')
forms = set()
# lemmas = set()  # not currently used
forms_stemmed = defaultdict(set)
with open(args.conllu_all) as conllufile:
    for line in conllufile:
        fields = line.split()
        if fields and fields[0].isdecimal():
            assert len(fields) > 2
            form = fields[1]
            #lemma = fields[2]
            # pos = fields[3]
            if args.lowercase:
                form = form.lower()
                #lemma = lemma.lower()
            if form in embedding:
                forms.add(form)
                forms_stemmed[get_stem(form)].add(form)
            #if lemma in embedding:
                #lemmas.add(lemma)
                #forms.add(lemma)
                #forms_stemmed[get_stem(lemma)].add(lemma)

logging.info('Read in test form-lemma pairs')
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
logging.info('Done reading')

def get_dist(form1, form2):
    # similarity to distance
    # if form1 != form2 and form1 in embedding and form2 in embedding:
    if form1 in embedding and form2 in embedding:
        return 1-similarity(form1, form2)
    else:
        return None


# list of indexes -> list of words
def node2str(node, index2word):
    return [index2word[index] for index in node]

def linkage(cluster1, cluster2, D):
    linkages = list()
    for node1 in cluster1:
        for node2 in cluster2:
            linkages.append(D[node1, node2])
    # min avg max
    return min(linkages), sum(linkages)/len(linkages), max(linkages)


# cluster each hypercluster
logging.info('Run the main loop')

L = 'average'
iterate_over = forms_stemmed
if args.plot:
    iterate_over = [args.plot]

# form -> cluster
result = defaultdict(str)

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

    C = max(int(I/3), 2)
    C = I

    clustering = AgglomerativeClustering(affinity='precomputed',
            linkage = L,
            compute_full_tree = True,
            n_clusters=C)

    logging.debug(stem)
    logging.debug(forms)
    logging.debug(I)
    logging.debug(C)
    logging.debug(dir(clustering))
    #logging.debug(D)


    # new branch -- cut off at given linkage
    if I == 1:
        result[index2word[0]] = stem + '0'
    else:
        clustering.fit(D)
        # default: each has own cluster
        clusters = list(range(I))
        # at the i-th iteration, children[i][0] and children[i][1] are merged to form node n_samples + i
        nodes = [[i] for i in range(I)]
        for merge in clustering.children_:
            # perform the merge
            node = list()
            node.extend(nodes[merge[0]])
            node.extend(nodes[merge[1]])
            nodes.append(node)
            # reassign words to new cluster ID
            for i in nodes[-1]:
                clusters[i] = len(nodes)
            # check stopping criterion
            lmin, lavg, lmax = linkage(nodes[merge[0]], nodes[merge[1]], D)
            if lavg > args.threshold:
                break
        for i, cluster in enumerate(clusters):
            result[index2word[i]] = stem + str(cluster)
            

    if args.plot:
        plt.title('Hierarchical Clustering Dendrogram')
        plot_dendrogram(clustering, labels=index2word)
        plt.show()

logging.info('Computing homogeneity.')
golden = list()
predictions = list()
for form, lemma in test_data:
    golden.append(lemma)
    predictions.append(result[form])
hcv = homogeneity_completeness_v_measure(golden, predictions)
print('Homogeneity', 'completenss', 'vmeasure', sep='\t')
print(*hcv, sep='\t')

logging.info('Done.')

