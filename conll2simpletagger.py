#!/usr/bin/env python3
#coding: utf-8

import pickle
import sys
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO)

logging.info("Read in forms set")
forms = set()
with open('syn_v4_forms_lemmas_50l.lc.vectors') as vecfile:
    for line in vecfile:
        fields = line.split()
        forms.add(fields[0].lower())
logging.info("Reading in {} forms done".format(len(forms)))

form2tags = defaultdict(Counter)
for line in sys.stdin:
    if line != '\n':
        form, lemma, tag = line.rstrip().split('\t')
        form = form.lower()
        if form in forms:
            if form not in form2tags and  len(form2tags) % 1000 == 0:
                logging.info("We now have {} forms".format(len(form2tags)))
            form2tags[form][tag] += 1

logging.info("We now have {} forms and we have finished".format(len(form2tags)))

result = dict()
for form in form2tags:
    result[form] = form2tags[form].most_common(1)[0]


logging.info("Conversion done")

with open(sys.argv[1], 'wb') as outfile:
    pickle.dump(result, outfile)

logging.info("Writewourt done")
