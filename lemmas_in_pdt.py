#!/usr/bin/env python3
#coding: utf-8

import sys
from collections import defaultdict, Counter
from sklearn import metrics

lemma_forms = defaultdict(Counter)
lemmas = list()
forms = list()
with open(sys.argv[1]) as conllufile:
    for line in conllufile:
        fields = line.split()
        if fields and fields[0].isdecimal():
            assert len(fields) > 2
            form = fields[1]
            lemma = fields[2]
            lemma_forms[lemma][form] += 1
            lemmas.append(lemma)
            forms.append(form)

for lemma in sorted(lemma_forms.keys()):
    print(lemma, lemma_forms[lemma].most_common())

print('lemma is form')
print('hcv', metrics.homogeneity_completeness_v_measure(lemmas, forms))
print('rs', metrics.adjusted_rand_score(lemmas, forms))

print('gold')
print(metrics.homogeneity_completeness_v_measure(lemmas, lemmas))
print(metrics.adjusted_rand_score(lemmas, lemmas))

import random
print('selecting random lemmas from the set of lemmas')
randoms = list()
selection = list(lemma_forms.keys())
for form in forms:
    randoms.append(random.choice(selection))
print(metrics.homogeneity_completeness_v_measure(lemmas, randoms))
print(metrics.adjusted_rand_score(lemmas, randoms))



print('something like accuracy -- the most frequent form of lemma is correct, rest is incorrect')
correct = 0
incorrect = 0
for lemma in lemma_forms.keys():
    mostcomon = lemma_forms[lemma].most_common(1)[0][1]
    allforms = sum(lemma_forms[lemma].values())
    correct += mostcomon
    incorrect += (allforms - mostcomon)
print(correct, incorrect, ((correct / (correct + incorrect))))


