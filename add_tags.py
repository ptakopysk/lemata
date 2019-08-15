#!/usr/bin/env python3
#coding: utf-8

import sys
import pickle

with open(sys.argv[2], 'rb') as dictfile:
    tagger = pickle.load(dictfile)

with open(sys.argv[1]) as infile:
    # strip headers
    infile.readline()
    infile.readline()
    for line in infile:
        line = line.rstrip('\n')
        items = line.split("\t")
        if (items[0].startswith('===')):
            pass
        else:
            form1 = items[4]
            form2 = items[5]
            tag1 = tagger[form1][0] if form1 in tagger else '????????????????'
            tag2 = tagger[form2][0] if form2 in tagger else '????????????????'
            print(*items, tag1, tag2, sep="\t")


