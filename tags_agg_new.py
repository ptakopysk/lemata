#!/usr/bin/env python3
#coding: utf-8

import sys
from collections import defaultdict, Counter
import statistics 

MINCOUNT = 1

# look for inflections where only 1 position is changed

pos2name = [ 'POS', 'SUBPOS', 'GENDER', 'NUMBER', 'CASE', 'POSSGENDER', 'POSSNUMBER', 'PERSON', 'TENSE', 'GRADE', 'NEGATION', 'VOICE', 'RESERVE1', 'RESERVE2', 'VAR',
'ASPECT', ]

change = defaultdict(list)
derchange = defaultdict(list)

def tag_distance(tag1, tag2):
    diff = 0
    for pos1, pos2 in zip(tag1, tag2):
        if pos1 != pos2 and pos1 != '-' and pos2 != '-':
            diff += 1
    return diff

for line in sys.stdin:
    fields = line.rstrip().split('\t')
    #count1 = fields[0]
    #count2 = fields[1]
    optp = fields[2]
    dist = float(fields[3])
    tag1 = fields[6]
    tag2 = fields[7]
    if tag1.startswith('???') or tag2.startswith('???'):
        # unknown tag
        continue
    if optp == 'INFL':
        # we are now only interested in inflections
        for index, (pos1, pos2) in enumerate(zip(tag1, tag2)):
            if pos1 != pos2:
                change[index].append(dist)
    else:
        derchange[optp].append(dist)

results = list()

for index in change:
    l = len(change[index])
    if l > MINCOUNT:
        sd = statistics.stdev(change[index])
        s = sum(change[index])
        avg = s/l
        results.append("\t".join(
            ("{:.4f}".format(avg), "{:.4f}".format(sd), pos2name[index], str(l))))
    else:
        print('Skipping', pos2name[index], str(l), file=sys.stderr)

for dertype in derchange:
    l = len(derchange[dertype])
    if l > MINCOUNT:
        sd = statistics.stdev(derchange[dertype])
        s = sum(derchange[dertype])
        avg = s/l
        results.append("\t".join(
            ("{:.4f}".format(avg), "{:.4f}".format(sd), dertype, str(l))))
    else:
        print('Skipping', dertype, str(l), file=sys.stderr)

results.sort()
print(*results, sep="\n")

