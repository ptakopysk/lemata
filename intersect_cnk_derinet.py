#!/usr/bin/env python3
#coding: utf-8

import sys
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

derinet_file, cnk_file = sys.argv[1:3]

logging.info("Read derinet")
number2lemma = dict()
number2parent = dict()
with open(derinet_file) as fh:
    for line in fh:
        # number, lemma, fulllemma, pos, parent = line.split()
        fields = line.split()
        number = int(fields[0])
        lemma = fields[1]
        if len(fields) == 4:
            parent = None
        else:
            parent = int(fields[4])
        number2lemma[number] = lemma
        number2parent[number] = parent

lemma2rootlemma = dict()
homonymous = set()
for number in number2lemma:
    # find root
    n = number
    while number2parent[n] != None:
        n = number2parent[n]
    rootlemma = number2lemma[n]
    
    lemma = number2lemma[number]
    if lemma in lemma2rootlemma:
        # seeing the lemma for a second time
        homonymous.add(lemma)
    lemma2rootlemma[lemma] = rootlemma

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

