#!/usr/bin/env python3
#coding: utf-8

import sys

from collections import defaultdict

from numpy import linspace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

filename = sys.argv[1]

for aggregate in (True, False):

    logging.info('PASS WITH aggregate = {}'.format(aggregate))

    logging.info('Read in data')
    dists = defaultdict(list)
    counts = defaultdict(list)
    with open(filename) as infile:
        for line in infile:
            count1, count2, optp, dist, _, _, _, _ = line.rstrip('\n').split('\t')
            dist = float(dist)
            count = int(count1)*int(count2)
            if aggregate:
                optp = optp[:4]
            dists[optp].append(dist)
            counts[optp].append(count)

    logging.info('Start plotting')
    plt.figure(figsize=(10,10), dpi=300)
    bins = 1000

    for optp in sorted(dists):
        logging.info('Plot {}'.format(optp))
        if optp.startswith('DERI'):
            ls = '-'
        elif optp in ('INFL_VOICE', 'INFL_NEGATION', 'INFL_GRADE', 'INFL_SUBPOS', 'INFL_TENSE'):
            ls = '--'
        else:
            ls = ':'
        _, bins, _ = plt.hist(dists[optp], bins, weights=counts[optp], density=True, histtype='step',
                cumulative=True, label=optp, ls=ls)
        
    logging.info('Set up graph')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.title(filename)
    plt.xlabel('Word form distance')
    plt.ylabel('Proportion of word form pairs, CDF')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(linspace(0, 1, 21))
    plt.yticks(linspace(0, 1, 21))

    agg = '.agg' if aggregate else ''
    outfilename = "cdf/" + filename + agg + ".weighted.png"
    logging.info('Save graph as ' + outfilename)
    plt.savefig(outfilename)

    logging.info('Done')
