#!/usr/bin/env python3
#coding: utf-8

import argparse
import sys
from collections import defaultdict

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(
        description='find lemma for form as nearest lemma in emb space')
ap.add_argument('filename',
        help='file with the embeddings')
ap.add_argument('conllu',
        help='file with the forms and lemmas')
ap.add_argument("-l", "--lemmadetection", type=str,
        help="Train lemma detection; 'svm' or 'logreg' or MLP eg '(100,)'")
ap.add_argument("-n", "--number", type=int,
        help="How many embeddings to read in")
ap.add_argument("-N", "--normalize", action="store_true",
        help="Normalize the embeddings")
ap.add_argument("-G", "--gold_stems", action="store_true",
        help="only look for lemmas with the right stem; gold stem used, i.e.  the longest common prefix of the form and the true lemma, also potentially removing 'ne-' prefix")
ap.add_argument("-s", "--similarity", type=str,
        help="Similarity: cos or jw")
args = ap.parse_args()

def similarity(word, otherword):
    sim = 0
    if ap.similarity == 'jw':
        sim = -distance.get_jaro_distance(word, otherword)
    if ap.similarity == 'jwxcos':
        emb1 = embedding[word]
        emb2 = embedding[otherword]
        cos_sim = inner(emb1, emb2)/(norm(emb1)*norm(emb2))
        jw_sim = 1-distance.get_jaro_distance(word, otherword)
        sim = jw_sim * cos_sim
    else:
        # cos
        emb1 = embedding[word]
        emb2 = embedding[otherword]
        sim = inner(emb1, emb2)/(norm(emb1)*norm(emb2))
    return sim

# read in embs
embedding = defaultdict(list)
with open(args.filename) as embfile:
    size, dim = map(int, embfile.readline().split())
    if args.number:
        size = min(size, args.number)
    for i in range(size):
        fields = embfile.readline().split()
        embedding[fields[0]] = list(map(float, fields[1:]))
        if args.normalize:
            embedding[fields[0]] /= norm(embedding[fields[0]])

# read in forms and lemmas
form2lemmas = defaultdict(set)
form2pos = dict()
with open(args.conllu) as conllufile:
    for line in conllufile:
        fields = line.split()
        if fields and fields[0].isdecimal():
            assert len(fields) > 2
            form = fields[1]
            lemma = fields[2]
            pos = fields[3]
            form2lemmas[form].add(lemma)
            form2pos[form] = pos
            form2pos[lemma] = pos
# remove ambiguous forms
form2lemma = dict()
lemmas_flat = set()
lemmas = defaultdict(set)
formsAndLemmas = set()
for form in form2lemmas:
    if len(form2lemmas[form]) > 1:
        print('Removing ambiguous form', form, '=', form2lemmas[form], file=sys.stderr)
    else:
        lemma = form2lemmas[form].pop()
        if form != lemma and form in embedding and lemma in embedding:
            form2lemma[form] = lemma
            pos = form2pos[lemma]
            lemmas[pos].add(lemma)
            lemmas_flat.add(lemma)
            formsAndLemmas.add(form)
            formsAndLemmas.add(lemma)


def find_gold_stem(form, lemma):
    stem = lemma
    while not form.startswith(stem):
        stem = stem[:-1]
    return stem

def gold_stems(form, lemma):
    stems = set()
    stems.add(find_gold_stem(form, lemma))
    form = form.lower()
    stems.add(find_gold_stem(form, lemma))
    if form.startswith('nej'):
        form = form[3:]
        stems.add(find_gold_stem(form, lemma))
    if form.startswith('ne'):
        form = form[2:]
        stems.add(find_gold_stem(form, lemma))
    stems.discard('')
    return stems

def check_stem(lemma, stems):
    if stems:
        for stem in stems:
            if lemma.startswith(stem):
                return True
        return False
    else:
        return True

if args.lemmadetection:
    # prepare data
    vecs = list()
    labels = list()
    for form in formsAndLemmas:
        islemma = form in lemmas_flat
        vecs.append(embedding[form])
        labels.append(islemma)
    X_train, X_test, y_train, y_test = train_test_split(
        vecs, labels, test_size=0.4, random_state=0)
    # train classifier
    if args.lemmadetection == 'svm':
        clf = svm.SVC(kernel='linear', C=1)
    elif args.lemmadetection == 'logreg':
        clf = linear_model.LogisticRegression()
    else:
        sizes = make_tuple(args.lemmadetection)
        clf = MLPClassifier(solver='adam', max_iter=500,
                hidden_layer_sizes=sizes)
    clf = clf.fit(X_train, y_train)
    # eval
    base = 1 - len(lemmas_flat)/len(formsAndLemmas)
    print('always predict False:', base)
    y_pred = clf.predict(X_train)
    trainacc = accuracy_score(y_train, y_pred)
    print(args.lemmadetection)
    print('test accuracy', 'train accuracy', sep="\t")
    print(clf.score(X_test, y_test), trainacc, sep="\t")
else:
    good = 0
    total = 0
    for form in form2lemma:
        best_sim = -2
        best_lemma = 'NONE'
        stems = None
        if args.gold_stems:
            stems = gold_stems(form, form2lemma[form])
        pos = form2pos[form]
        for lemma in lemmas[pos]:
            if check_stem(lemma, stems):
                sim = similarity(form, lemma)
                if sim > best_sim:
                    best_sim = sim
                    best_lemma = lemma
        
        ok = best_lemma == form2lemma[form]
        print(form, '->', best_lemma, round(best_sim, 2), ok)
        total += 1
        if ok:
            good += 1
    
    print('RESULT:', good, '/', total, '=', round(good/total*100,2), '%')




