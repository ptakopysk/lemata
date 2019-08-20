#!/usr/bin/env python3
#coding: utf-8

import sys
from collections import defaultdict, Counter

# look for inflections where only 1 position is changed

pos2name = [ 'POS', 'SUBPOS', 'GENDER', 'NUMBER', 'CASE', 'POSSGENDER', 'POSSNUMBER', 'PERSON', 'TENSE', 'GRADE', 'NEGATION', 'VOICE', 'RESERVE1', 'RESERVE2', 'VAR',
'ASPECT', ]

change = defaultdict(list)

for line in sys.stdin:
    fields = line.rstrip().split('\t')
    optp = fields[2]
    dist = float(fields[3])
    tag1 = fields[6]
    tag2 = fields[7]
    if tag1.startswith('???') or tag2.startswith('???'):
        # unknown tag
        continue
    if optp == 'DERI':
        # we are now only interested in inflections
        continue

    diff = 0
    for pos1, pos2 in zip(tag1, tag2):
        if pos1 != pos2:
            diff += 1
    #assert diff > 0
    if diff == 1:
        # tags differ in just one value, this is what we are looking for
        print(line.rstrip())

