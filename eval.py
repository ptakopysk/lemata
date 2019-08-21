#!/usr/bin/env python3
#coding: utf-8

import sys

import logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

logging.info('Read in data')
results = list()
count_deri = 0
count_infl = 0
for line in sys.stdin:
    count1, count2, optp, dist, form1, form2, tag1, tag2 = line.rstrip('\n').split('\t')
    if optp.startswith('DERI'):
        count_deri += 1
    else:
        assert optp.startswith('INFL')
        count_infl += 1
    results.append((dist, count_deri, count_infl))

total_count_infl = count_infl
total_count_deri = count_deri

def f(p, r):
    if p > 0 and r > 0:
        return 2 * p * r / (p + r)
    else:
        return 0

logging.info('Compute P R F')

best_f_deri = 0
best_dist_deri = 0
best_f_infl = 0
best_dist_infl = 0

for dist, count_deri, count_infl in results:
    p_deri = count_deri / (count_deri + count_infl)
    r_deri = count_deri / total_count_deri
    f_deri = f(p_deri, r_deri)
    if f_deri > best_f_deri:
        best_f_deri = f_deri
        best_dist_deri = dist

    p_infl = count_infl / (count_infl + count_infl)
    r_infl = count_infl / total_count_infl
    f_infl = f(p_infl, r_infl)
    if f_infl > best_f_infl:
        best_f_infl = f_infl
        best_dist_infl = dist

logging.info('Print result')
print('For infl, best F1 = {} achieved with dist = {}'.format(best_f_infl, best_dist_infl))
print('For deri, best F1 = {} achieved with dist = {}'.format(best_f_deri, best_dist_deri))


