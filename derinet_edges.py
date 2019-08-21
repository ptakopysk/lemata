#!/usr/bin/env python3
#coding: utf-8

import sys

import logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

logging.info('Read in Derinet')
derinet_lemma = dict()
derinet_parent = dict()
with open('derinet-1-7.tsv') as infile:
    for line in infile:
        fields = line.rstrip('\n').split('\t')
        if fields:
            assert len(fields) == 5, line
            did = fields[0]
            lemma = fields[1]
            pos = fields[3]
            parentid = fields[4]
            derinet_lemma[did] = (lemma, pos)
            if parentid.isdecimal():
                derinet_parent[did] = parentid

logging.info('Compute Derinet roots')
def find_root(did):
    while did in derinet_parent:
        did = derinet_parent[did]
    return did
derinet_root = {did:find_root(did) for did in derinet_lemma}

logging.info('Read in top 50 roots')
top_roots = set()
with open('derroot_lemma_forms_50l.roots.top50') as infile:
    for line in infile:
        top_roots.add(line.rstrip())

logging.info('Write out lemmas of the top 50 roots')
for did in derinet_lemma:
    lemma, pos = derinet_lemma[did]
    root_did = derinet_root[did]
    root_lemma, root_pos = derinet_lemma[root_did]
    if root_lemma in top_roots:
        if did in derinet_parent:
            parent_did = derinet_parent[did]
            parent_lemma, parent_pos = derinet_lemma[parent_did]
        else:
            parent_lemma = ''
            parent_pos = ''
        print(root_lemma, lemma, pos, parent_lemma, parent_pos, sep="\t")

logging.info('Done')

