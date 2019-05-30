#!/usr/bin/env python3
#coding: utf-8

from collections import defaultdict
import sys

from pyjarowinkler import distance


# Jaro Winkler that can take emtpy words
def jw_safe(srcword, tgtword):
    if srcword == '' or tgtword == '':
        # 1 if both empty
        # 0.5 if one is length 1
        # 0.33 if one is length 2
        # ...
        return 1/(len(srcword)+len(tgtword)+1)
    elif srcword == tgtword:
        return 1
    else:
        return distance.get_jaro_distance(srcword, tgtword)

def jwsim(word, otherword):
    # called distance but is actually similarity
    sim = jw_safe(word, otherword)
    uword = devow(word)
    uotherword = devow(otherword)
    usim = jw_safe(uword, uotherword)    
    sim = (sim+usim)/2
    assert sim >= 0 and sim <= 1, "JW sim must be between 0 and 1"
    return sim




root_lemma_forms = defaultdict(dict)
for line in sys.stdin:
    words = line.split()
    root = words[0]
    lemma = words[1]
    forms = words[1:]  # the lemma is also a form
    root_lemma_forms[root][lemma] = forms

for root in root_lemma_forms:
# TODO cluster all the forms of all the lemmas
# the lemmas are the cluter labels
# number of lemmas is number of clusters
# evaluate


