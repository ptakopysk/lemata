#!/usr/bin/env python3
#coding: utf-8

import sys

from collections import defaultdict

import logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

#MINCOUNT = 10
MINCOUNT = 1

logging.info('Read in Derinet edges')
lemma2plemma = dict()
with open('derinet_edges.tsv') as infile:
    for line in infile:
        root_lemma, lemma, pos, parent_lemma, parent_pos = line.rstrip('\n').split('\t')
        lemma2plemma[(lemma.lower(), pos)] = (parent_lemma.lower(), parent_pos)

logging.info('Read in lemma forms')
lemma2forms = defaultdict(set)
with open('lemma_forms.tsv') as infile:
    for line in infile:
        lemma, pos, form, tag, count = line.rstrip('\n').split("\t")
        lemma2forms[(lemma.lower(), pos)].add((form.lower(), tag, count))

printbuffer = set()

def printout(inde, form1, form2):
    if form1[0] == form2[0]:
        return

    if int(form1[2]) < MINCOUNT or int(form2[2]) < MINCOUNT:
        return

    if form1[0] > form2[0]:
        tmp = form2
        form2 = form1
        form1 = tmp
        
    printbuffer.add((inde, *form1, *form2))

def tag_distance(form1, form2):
    return _tag_distance(form1[1], form2[1])

def _tag_distance(tag1, tag2):
    diff = 0
    for index, (pos1, pos2) in enumerate(zip(tag1, tag2)):
        # different value
        # is not subpos
        # is not unspecified
        if pos1 != pos2 and index != 1 and pos1 != '-' and pos2 != '-':
            diff += 1
    return diff

pos2name = [ 'POS', 'SUBPOS', 'GENDER', 'NUMBER', 'CASE', 'POSSGENDER', 'POSSNUMBER', 'PERSON', 'TENSE', 'GRADE', 'NEGATION', 'VOICE', 'RESERVE1', 'RESERVE2', 'VAR', 'ASPECT', ]

def optypes(form1, form2):
    optypes = list()
    tag1 = form1[1]
    tag2 = form2[1]
    for index, (pos1, pos2) in enumerate(zip(tag1, tag2)):
        if pos1 != pos2:
            optypes.append(pos2name[index])
    return optypes

logging.info('Prepare pairs')
for lemma, pos in lemma2forms:
    forms = lemma2forms[(lemma, pos)]

    # inflections
    for form1 in forms:
        for form2 in forms:
            if tag_distance(form1, form2) <= 1:
                for optp in optypes(form1, form2):
                    printout('INFL_' + optp, form1, form2)

    # derivations
    if (lemma, pos) in lemma2plemma:
        (plemma, ppos) = lemma2plemma[(lemma, pos)]
        if (plemma, ppos) in lemma2forms:  # plemma may be OOV
            pforms = lemma2forms[(plemma, ppos)]
            for form, tag, count in forms:
                for pform, ptag, pcount in pforms:
                    if tag == ptag:
                        printout('DERI_TAG',
                                (form, tag, count), (pform, ptag, pcount))
                    elif form == lemma and pform == plemma:
                        printout('DERI_EDGE',
                                (form, tag, count), (pform, ptag, pcount))

logging.info('Print out pairs')
for item in printbuffer:
    print(*item, sep='\t')

logging.info('Done')
