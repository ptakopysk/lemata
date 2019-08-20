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

    diff = list()
    for index, (pos1, pos2) in enumerate(zip(tag1, tag2)):
        if pos1 != pos2:
            #diff.append((index, pos1, pos2))
            diff.append((index))
    if len(diff) == 1:
        # tags differ in just one value, this is what we are looking for
        #index, pos1, pos2 = diff[0]
        #pos1, pos2 = sorted((pos1, pos2))
        #change[(index, pos1, pos2)].append(dist)        
        index = diff[0]
        change[index].append(dist)        

#for index, pos1, pos2 in change:
for index in change:
    #s = sum(change[(index, pos1, pos2)])
    #l = len(change[(index, pos1, pos2)])
    s = sum(change[index])
    l = len(change[index])
    avg = s/l
    #print("{:.4f}".format(avg), pos2name[index], pos1, pos2, l, sep="\t")
    print("{:.4f}".format(avg), pos2name[index], l, sep="\t")

