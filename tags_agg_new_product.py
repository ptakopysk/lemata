#!/usr/bin/env python3
#coding: utf-8

import sys
from collections import defaultdict, Counter
import statistics 

MINCOUNT = 1

# look for inflections where only 1 position is changed

pos2name = [ 'POS', 'SUBPOS', 'GENDER', 'NUMBER', 'CASE', 'POSSGENDER', 'POSSNUMBER', 'PERSON', 'TENSE', 'GRADE', 'NEGATION', 'VOICE', 'RESERVE1', 'RESERVE2', 'VAR', 'ASPECT', ]

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
    count1 = int(fields[0])
    count2 = int(fields[1])
    count = count1 * count2
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
                change[index].append((dist, count))
    else:
        derchange[optp].append((dist, count))

results = list()

for index in change:
    l = sum([c[1] for c in change[index]])
    if l > MINCOUNT:
        #sd = statistics.stdev(change[index])
        sd = 1
        s = sum([c[0] * c[1] for c in change[index]])
        avg = s/l
        results.append("\t".join(
            ("{:.4f}".format(avg), "{:.4f}".format(sd), pos2name[index], str(l))))
    else:
        print('Skipping', pos2name[index], str(l), file=sys.stderr)

for dertype in derchange:
    l = sum([c[1] for c in derchange[dertype]])
    if l > MINCOUNT:
        #sd = statistics.stdev(derchange[dertype])
        sd = 1
        s = sum([c[0] * c[1] for c in derchange[dertype]])
        avg = s/l
        results.append("\t".join(
            ("{:.4f}".format(avg), "{:.4f}".format(sd), dertype, str(l))))
    else:
        print('Skipping', dertype, str(l), file=sys.stderr)

results.sort()
print(*results, sep="\n")

