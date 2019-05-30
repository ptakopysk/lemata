#!/usr/bin/env python3
#coding: utf-8

import sys
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

derinet_file, cnk_file = sys.argv[1:3]

logging.info("Read derinet")
number2rootlemma = dict()
lemma2rootlemma = dict()
homonymous = set()
with open(derinet_file) as fh:
    for line in fh:
        fields = line.split()
        # number, lemma, fulllemma, pos, parent = line.split()
        number = int(fields[0])
        lemma = fields[1]
        if lemma in lemma2rootlemma:
            # seeing the lemma for a second time
            homonymous.add(lemma)
            
        if len(fields) == 4:
            # no parent -> is root
            rootlemma = lemma
        else:
            parent = int(fields[4])
            rootlemma = number2rootlemma[parent]
        lemma2rootlemma[lemma] = rootlemma
        number2rootlemma[number] = rootlemma
logging.info("Read {} derinet lemmas".format(len(lemma2rootlemma)))

logging.info("Remove homonyms")
for lemma in homonymous:
    del lemma2rootlemma[lemma]
logging.info("Removed {} homonyms".format(len(homonymous)))

logging.info("Read CNK")
rootlemmaforms = defaultdict(list)
with open(cnk_file) as fh:
    for line in fh:
        words = line.split()
        lemma = words[0]
        if lemma in lemma2rootlemma:
            rootlemma = lemma2rootlemma[lemma]
            rootlemmaforms[rootlemma].append(words)
logging.info("Read in forms for {} roots".format(len(rootlemmaforms)))

logging.info("Write intersection")
for rootlemma in rootlemmaforms:
    if len(rootlemmaforms[rootlemma]) > 1:
        for words in rootlemmaforms[rootlemma]:
            print(rootlemma, *words)

logging.info("Done")

