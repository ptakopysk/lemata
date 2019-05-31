#!/usr/bin/env python3
#coding: utf-8

import sys
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

embeddings_file, data_file = sys.argv[1:3]

vocabulary = set()
with open(data_file) as fh:
    for line in fh:
        words = line.lower().split()
        vocabulary.update(words)

backup = list()

n = 0
with open(embeddings_file) as fh:
    for line in fh:
        n += 1
        word = line.split(' ', 1)[0]
        if word in vocabulary:
            print(line, end='')
            vocabulary.remove(word)
        elif word.lower() in vocabulary:
            backup.append(line)
        if n%10000 == 0:
            logging.info("Read {} lines".format(n))

if len(vocabulary) > 0:
    logging.warning("{} items not covered by the embeddings!!!".format(len(vocabulary)))
    for word in vocabulary:
        logging.info("Not covered: {}".format(word))

    logging.warning("TRYING TO USE THE BACKUP")
    for line in backup:
        l = line.lower()
        word = l.split(' ', 1)[0]
        if word in vocabulary:
            print(l, end='')
            vocabulary.remove(word)
    if len(vocabulary) > 0:
        logging.warning("{} items not covered by the embeddings even as backup!!!".format(len(vocabulary)))
        for word in vocabulary:
            logging.info("Not covered by backup: {}".format(word))

