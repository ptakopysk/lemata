#!/usr/bin/env python3
#coding: utf-8

import argparse
import sys
from collections import defaultdict

from sklearn.metrics import accuracy_score
from numpy import inner
from numpy.linalg import norm

from pyjarowinkler import distance

import unidecode

ap = argparse.ArgumentParser(
        description='compare distance of lemma to its inflections and to its derivation(s)')
ap.add_argument('filename',
        help='file with the embeddings')
ap.add_argument('conllu',
        help='file with the forms and lemmas')
ap.add_argument('derinet',
        help='file with the derivations')
ap.add_argument("-n", "--number", type=int,
        help="How many embeddings to read in")
ap.add_argument("-N", "--normalize", action="store_true",
        help="Normalize the embeddings")
ap.add_argument("-s", "--similarity", type=str,
        help="Similarity: cos or jw or jwxcos or jwxcosxlen")
ap.add_argument("-l", "--length", type=float, default=0.05,
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
    else:
        # cos
        return embsim(word, otherword)

print('Read in embeddings', file=sys.stderr)
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

print('Read in forms and lemmas', file=sys.stderr)
lemma2forms = defaultdict(set)
with open(args.conllu) as conllufile:
    for line in conllufile:
        fields = line.split()
        if fields and fields[0].isdecimal():
            assert len(fields) > 2
            form = fields[1]
            lemma = fields[2]
            if form != lemma:
                # cases when form==lemma are trivial and uninteresting
                lemma2forms[lemma].add(form)

print('Read in Derinet', file=sys.stderr)
derinet_lemma = dict()
derinet_parent = dict()
with open(args.derinet) as derinetfile:
    for line in derinetfile:
        fields = line.rstrip('\n').split('\t')
        if fields:
            assert len(fields) == 5, line
            did = fields[0]
            lemma = fields[1]
            # pos = fields[3]
            parentid = fields[4]
            derinet_lemma[did] = lemma
            if parentid.isdecimal():
                derinet_parent[did] = parentid
print('Done reading', file=sys.stderr)

good = 0
total = 0
for did, lemma in derinet_lemma.items():
    if lemma in lemma2forms and lemma in embedding and did in derinet_parent:
        parent = derinet_lemma[derinet_parent[did]]
        forms = lemma2forms[lemma]
        if parent in forms:
            # print(parent, 'is both a parent and a form of', lemma, file=sys.stderr)
            pass
        elif parent in embedding:
            forms_int = forms & embedding.keys()
            if len(forms_int) > 0:
                parent_sim = similarity(lemma, parent)
                print('DERPARENT', lemma, '->', parent, ':', parent_sim)
                for form in forms_int:
                    form_sim = similarity(lemma, form)
                    ok = form_sim > parent_sim
                    print('FORM', lemma, '->', form, ':', form_sim, ok)
                    total += 1
                    if ok:
                        good += 1

print('RESULT:', good, '/', total, '=', round(good/total*100,2), '%')




