#!/usr/bin/env python3
#coding: utf-8

import argparse
import sys
from collections import defaultdict
from sortedcollections import ValueSortedDict

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

from pyjarowinkler import distance

import unidecode

import matplotlib
matplotlib.use('Agg')
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
ap.add_argument("-N", "--normalize", action="store_true",
        help="Normalize the embeddings")
ap.add_argument("-s", "--similarity", type=str,
        help="Similarity: cos or jw")
ap.add_argument("-L", "--length", type=float, default=0.05,
        help="Weight for length similarity")
args = ap.parse_args()


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

print('Read in forms and lemmas', file=sys.stderr)
forms = set()
lemmas = set()
with open(args.conllu_all) as conllufile:
    for line in conllufile:
        fields = line.split()
        if fields and fields[0].isdecimal():
            assert len(fields) > 2
            form = fields[1]
            lemma = fields[2]
            # pos = fields[3]
            forms.add(form)
            forms.add(lemma)
            lemmas.add(lemma)

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
            test_data.append((form, lemma))
print('Done reading', file=sys.stderr)

print('Compute all similarities', file=sys.stderr)
sims = dict()
for form1 in forms:
    if form1 not in embedding:
        continue
    sims[form1] = ValueSortedDict()
    for form2 in forms:
        if form2 not in embedding:
            continue
        sims[form1][form2] = similarity(form1, form2)
print('Done computing similarities', file=sys.stderr)

def cluster_lemma(cluster):
    result = None
    for form in cluster:
        if form in lemmas:
            result = form
    return result

good = 0
total = 0
for form, lemma in test_data:
    if form in lemmas:
        continue
    if form not in embedding:
        continue
    
    total += 1
    cluster = {form}
    while cluster_lemma(cluster) == None:
        # TODO find nearest going over all forms in cluster!!!
        # now finds nearest lemma to form, ie should be identical to what I
        # have (but without various filterins)
        _, next_form = sims[form].peekitem(len(cluster) - 1)
        cluster.add(next_form)
    if cluster_lemma(cluster) == lemma:
        good += 1
    
print('RESULT:', good, '/', total, '=', round(good/total*100,2), '%')




