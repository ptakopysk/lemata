#!/usr/bin/env python3
#coding: utf-8

lemma2rootlemma = dict()
with open('derinet_neighbours.tsv') as neifile:
    for line in neifile:
        fields = line.rstrip().split('\t')
        lemma = fields[0]
        rootlemma = fields[1]
        lemma2rootlemma[lemma] = rootlemma
    
with open('derroot_lemma_forms_50l.ssv') as ssvfile:
    for line in ssvfile:
        fields = line.rstrip().split(' ')
        root = fields[0]
        lemma = fields[1]
        forms = fields[2:]
        if lemma in lemma2rootlemma:
            rootlemma = lemma2rootlemma[lemma]
        else:
            rootlemma = ''
        print(root, lemma, rootlemma, *forms, sep='\t')


