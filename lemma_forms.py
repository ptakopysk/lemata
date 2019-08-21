#!/usr/bin/env python3
#coding: utf-8

import sys
import gzip
from collections import defaultdict, Counter

import logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

logging.info('Read in Derinet edges')
lemmas = set()
with open('derinet_edges.tsv') as infile:
    for line in infile:
        root_lemma, lemma, pos, parent_lemma, parent_pos = line.rstrip('\n').split('\t')
        lemmas.add((lemma, pos))

logging.info('Read in SYN')
result = Counter()
lines = 0
with gzip.open('syn_v4.conll.gz', 'rt') as infile:
    for line in infile:
        fields = line.rstrip().split('\t')
        if len(fields) == 3:
            form, lemma, tag = fields
            pos = tag[0]
            if (lemma, pos) in lemmas:
                result[(lemma, pos, form, tag)] += 1
        lines += 1
        if (lines % 1000000 == 0):
            logging.info("{}M lines read".format(lines/1000000))
        #if (lines % 10000000 == 0):
        #    break

logging.info('Produce output')
for lemma, pos, form, tag in result:
    count = result[(lemma, pos, form, tag)]
    print(lemma, pos, form, tag, count, sep='\t')

logging.info('Done')
