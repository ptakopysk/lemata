#!/usr/bin/env python3
#coding: utf-8

import sys
import gzip
from unidecode import unidecode
from collections import defaultdict
import logging

logging.basicConfig(level=logging.DEBUG)

# index:
# - 1-36: the corresponding letter
# - 37: other
letter_index = sys.argv[1]

# index 0 is not used
letters = "aabcdefghijklmnopqrstuvwxyz0123456789"
letters_set = set(letters)

letter_index = int(letter_index)
assert letter_index >= 1
assert letter_index <= 37
if letter_index < len(letters):
    letter = letters[letter_index]
else:
    letter = None

lemma2forms = defaultdict(set)
for line in sys.stdin:
    fields = line.split()
    if len(fields) >= 2:
        form = fields[0]
        lemma = fields[1]
        # asciized and lowercased
        initial = unidecode(lemma[0]).lower()
        if letter:
            # looking for alpha
            if letter == initial:
                # matches what we are searching
                lemma2forms[lemma].add(form)
        else:
            # looking for non-aplha
            if initial not in letters_set:
                # is non-alpha
                lemma2forms[lemma].add(form)

outputfile = 'lemma_forms/' + str(letter)
with open(outputfile, mode="w") as ofh:
    for lemma in lemma2forms:
        print(lemma, *lemma2forms[lemma], file=ofh)

