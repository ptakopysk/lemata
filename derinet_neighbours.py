#!/usr/bin/env python3
#coding: utf-8

import sys

print('Read in Derinet', file=sys.stderr)
derinet_lemma = dict()
derinet_parent = dict()
for line in sys.stdin:
    fields = line.rstrip('\n').split('\t')
    if fields:
        assert len(fields) == 5, line
        did = fields[0]
        lemma = fields[1]
        parentid = fields[4]
        derinet_lemma[did] = lemma
        if parentid.isdecimal():
            derinet_parent[did] = parentid
print('Done reading', file=sys.stderr)

for did in derinet_parent:
    lemma = derinet_lemma[did]
    parentlemma = derinet_lemma[derinet_parent[did]]
    print(lemma, parentlemma, sep="\t")
print('Done', file=sys.stderr)

