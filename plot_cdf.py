#!/usr/bin/env python3
#coding: utf-8

import sys

from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)


#neighbour_pairs_new_dist_dertype_jwxcos.tsv

logging.info('Read in data')
dists = defaultdict(list)
for line in sys.stdin:
    _, _, optp, dist, _, _, _, _ = line.rstrip('\n').split('\t')
    dist = float(dist)
    dists[optp].append(dist)

logging.info('Start plotting')
maxcount = max([len(dists[optp]) for optp in dists])
bins = maxcount
#bins = 100

for optp in sorted(dists):
    logging.info('Plot {}'.format(optp))
    ls = '-' if optp.startswith('DERI') else ':'
    _, bins, _ = plt.hist(dists[optp], bins, density=True, histtype='step',
            cumulative=True, label=optp, ls=ls)
    
logging.info('Set up graph')
#ax.grid(True)
plt.legend(loc='upper left')
plt.title('Empirical CDF')
plt.xlabel('Word form distance threshold')
plt.ylabel('Proportion of word form pairs, CDF')
plt.xticks([bins[0], bins[-1]])
#plt.xlim(0, maxcount)
#plt.ylim(0, 1)

#print(*bins)

logging.info('Save graph')
plt.savefig("cdf2.png")

logging.info('Done')
