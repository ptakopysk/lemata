#!/usr/bin/env python3
#coding: utf-8

import sys
#from unidecode import unidecode

for line in sys.stdin:
    words = line.split()
    if words[0].isalpha():
        result = [word for word in words if word.isalpha()]
        print(*result)



