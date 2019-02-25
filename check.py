#!/usr/bin/env python3
#coding: utf-8


from pyjarowinkler import distance

import sys

for word, otherword in [
        (sys.argv[1], sys.argv[2]),
        (sys.argv[1], sys.argv[3]),
        (sys.argv[2], sys.argv[3]),
        ]:
    sim = -distance.get_jaro_distance(word, otherword)
    print(word, otherword, sim)
    




