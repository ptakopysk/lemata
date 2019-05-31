#!/usr/bin/env python3
#coding: utf-8

import sys
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

data_file, size = sys.argv[1:3]
size=int(size)

root = None
lines = list()
with open(data_file) as fh:
    for line in fh:
        words = line.split()
        if words[0] != root:
            # moving to a new root
            if len(lines) > size:
                logging.info('Printing {} lemmas for root "{}"'.format(len(lines), root))
                for line in lines:
                    print(line, end='')
            lines = [line]
            root = words[0]
        else:
            lines.append(line)

