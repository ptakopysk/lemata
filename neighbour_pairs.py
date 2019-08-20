#!/usr/bin/env python3
#coding: utf-8

import sys
import pickle

with open('syn_v4.simpletagger.50l.lc.pickle', 'rb') as dictfile:
    tagger = pickle.load(dictfile)

lemma2forms = dict()
lemma2plemma = dict()
with open('derroot_lemma_forms_50l.ssv_withderparents') as infile:
    for line in infile:
        line = line.rstrip('\n')
        items = line.split("\t")
        root = items[0]
        lemma = items[1].lower()
        plemma = items[2].lower()
        
        forms = set([item.lower() for item in items[3:]])
        forms.add(lemma)
        lemma2forms[lemma] = forms
        if plemma:
            lemma2plemma[lemma] = plemma

def tag(form):
    return tagger[form][0] if form in tagger else '????????????????'

def printout(inde, form1, form2):
    if form1 != form2 and form1 in tagger:
        if form1 < form2:
            print(inde, form1, form2, tag(form1), tag(form2), sep="\t")
        else:
            print(inde, form2, form1, tag(form2), tag(form1), sep="\t")

def tag_distance(form1, form2):
    diff = 0
    for index, (pos1, pos2) in enumerate(zip(tag(form1), tag(form2))):
        # different value
        # is not subpos
        # is not unspecified
        if pos1 != pos2 and index != 1 and pos1 != '-' and pos2 != '-':
            diff += 1
    return diff

for lemma in lemma2forms:
    forms = lemma2forms[lemma]

    # inflections
    for form1 in forms:
        for form2 in forms:
            if tag_distance(form1, form2) <= 1:
                printout('INFL', form1, form2)

    # derivations
    if lemma in lemma2plemma:
        plemma = lemma2plemma[lemma]
        if plemma in lemma2forms:  # plemma may be OOV
            pforms = lemma2forms[plemma]
            printout('DERI', lemma, plemma)
            for form in forms:
                for pform in pforms:
                    if tag(form) == tag(pform):
                        printout('DERI', form, pform)


