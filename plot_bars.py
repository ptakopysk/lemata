#!/usr/bin/env python3
#coding: utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from collections import defaultdict

# dist
ind_value = list()
# infl/der
ind_is_infl = list()

with open(sys.argv[1]) as infile:
    # strip headers
    infile.readline()
    infile.readline()
    for line in infile:
        line = line.rstrip('\n')
        items = line.split("\t")
        if (items[0].startswith('===')):
            ind_value.append(1.0)
            ind_is_infl.append('r')
        else:
            dist = float(items[3])
            is_infl = 'b' if items[2] == 'INFL' else 'y'
            ind_value.append(dist)
            ind_is_infl.append(is_infl)

plt.figure(1)

#s = np.argsort(ind_value)
#x = np.arange(len(ind_value))
#plt.bar(x, height=[ind_value[k] for k in s], color=[ind_is_infl[k] for k in s])

r = range(len(ind_value))
plt.bar(r, height=ind_value, color=ind_is_infl)

#plt.show()
plt.savefig(sys.argv[1] + '.pdf')

