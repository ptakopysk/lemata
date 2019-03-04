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
ap.add_argument("-f", "--formbase", action="store_true",
        help="Base the search only on the original form")
ap.add_argument("-V", "--verbose", action="store_true",
        help="Print more verbose progress info")
ap.add_argument("-N", "--normalize", action="store_true",
        help="Normalize the embeddings")
ap.add_argument("-s", "--similarity", type=str,
        help="Similarity: cos or jw")
ap.add_argument("-S", "--stems", action="store_true",
        help="only look for words with the same stem")
ap.add_argument("-L", "--length", type=float, default=0.05,
        help="Weight for length similarity")
ap.add_argument("-C", "--cut", type=int, default=100,
        help="Cut down the number of most similar words to 100 for each word")
ap.add_argument("-U", "--upperbound", action="store_true",
        help="Highest achievable accuracy given the settings (esp. stem pruning)")
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

def get_stem(form):
    return form[:2]

print('Read in forms and lemmas', file=sys.stderr)
forms = set()
lemmas = set()
forms_stemmed = defaultdict(set)
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
            forms_stemmed[get_stem(form)].add(form)
            forms_stemmed[get_stem(lemma)].add(lemma)

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

def get_dist(form1, form2):
    # similarity to distance
    if form1 != form2 and form1 in embedding and form2 in embedding:
        return -similarity(form1, form2)
    else:
        return None

print('Compute all similarities', file=sys.stderr)
# distances instead of similarities for lowest distances to come first in
# natural ordering
dists = dict()
for form1 in forms:
    if form1 not in embedding:
        continue
    # compute all
    alldists = ValueSortedDict()
    iter_forms = None
    if args.stems:
        iter_forms = forms_stemmed[get_stem(form1)]
    else:
        iter_forms = forms
    for form2 in iter_forms:
        dist = get_dist(form1, form2)
        if dist is not None:
            alldists[form2] = dist
    # store top C
    dists[form1] = ValueSortedDict()
    for form2 in alldists.keys()[:args.cut]:
        dists[form1][form2] = alldists[form2]
    # progress
    if len(dists) % 1000 == 0:
        print(len(dists), file=sys.stderr)
print('Done computing similarities', file=sys.stderr)

def cluster_lemma(cluster):
    lemma = cluster.intersection(lemmas)
    if len(lemma) == 1:
        return lemma.pop()
    else:
        assert len(lemma) == 0
        return None

def find_lemma_agg(form):
    # find nearest going over all forms in cluster
    cluster = {form}
    while cluster_lemma(cluster) == None:
        next_form = None
        next_dist = 2  # distances are -1..1 so 2 is infinity
        for cluster_form in cluster:
            for cand_form, cand_dist in dists[cluster_form].items():
                # find first not in cluster yet
                if cand_form not in cluster:
                    # must be better than what we currently have
                    if cand_dist < next_dist:
                        next_form = cand_form
                        next_dist = cand_dist
                        if args.verbose:
                            print(next_form, next_dist, file=sys.stderr)
                    # break anyway
                    break                
        if next_form == None:
            # did not find anything, nothing more to do here
            return None
        else:
            cluster.add(next_form)
            if args.verbose:
                print(next_form, file=sys.stderr)
    return cluster_lemma(cluster)

def find_lemma_formbase(form):
    # finds nearest lemma to form, ie should be identical to what I
    # have (but without various filterins)
    for cand_form in dists[form]:
        if args.verbose:
            print(cand_form, file=sys.stderr)
        if cand_form in lemmas:
            return cand_form
    return None

def get_sim(form1, form2):
    if form1 in dists and form2 in dists[form1]:
        # distance to similarity
        return -dists[form1][form2]
    else:
        return -2

good = 0
total = 0
for form, lemma in test_data:
    if args.verbose:
        print(form, lemma, file=sys.stderr)
    if form in lemmas:
        continue
    if form not in embedding:
        continue
    
    if args.upperbound:
        # TODO for agg I should look whether there is path within dists...
        found_lemma = lemma if lemma in dists[form] else None
    elif args.formbase:
        found_lemma = find_lemma_formbase(form)
    else:
        found_lemma = find_lemma_agg(form)
        
    ok = (found_lemma == lemma)
    print(form, '->', found_lemma, round(get_sim(form, found_lemma), 4), ok)
    total += 1
    if ok:
        good += 1
    else:
        print('GOLD:', form, '->', lemma, round(get_sim(form, lemma), 4))

    
print('RESULT:', good, '/', total, '=', round(good/total*100,2), '%')




