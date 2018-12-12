#!/usr/bin/env python3
#coding: utf-8

import argparse

from collections import defaultdict

from numpy import inner
from numpy.linalg import norm

ap = argparse.ArgumentParser(description='show embedding neighbours')
ap.add_argument('filename',
        help='file with the embeddings')
ap.add_argument("-n", "--number", type=int,
        help="How many embeddings to read in")
ap.add_argument("-m", "--max", type=int, default=100,
        help="Only print out the 100 nearest neighbours")
ap.add_argument("-N", "--normalize", action="store_true",
        help="Normalize the embeddings")
args = ap.parse_args()

# read in embs
# TODO normalize?
embedding = defaultdict(list)
with open(args.filename) as embfile:
    size, dim = map(int, embfile.readline().split())
    if args.number:
        size = min(size, args.number)
    for i in range(size):
        fields = embfile.readline().split()
        embedding[fields[0]] = list(map(float, fields[1:]))
        if args.normalize:
            embedding[fields[0]] /= norm(embedding[fields[0]])

def similarity(word, otherword):
    emb1 = embedding[word]
    emb2 = embedding[otherword]
    return inner(emb1, emb2)/(norm(emb1)*norm(emb2))

while True:
    line = input("word [stem [stem ...]]: ")
    fields = line.split()
    
    # empty input
    if len(fields) == 0:
        continue

    word = fields[0]
    if word not in embedding:
        print('OOV')
        continue

    if len(fields) == 1:
        stems = [word[:5]]
    else:
        stems = fields[1:]
    print(word, stems)

    neighbours = set()
    for neighbour in embedding.keys():
        for stem in stems:
            if neighbour.startswith(stem):
                neighbours.add(neighbour)

    nearest = dict()
    for neighbour in neighbours:
        sim = similarity(word, neighbour)
        nearest[neighbour] = sim

    nearest_sorted = sorted(nearest, key=nearest.get, reverse=True)

    for word in nearest_sorted[:args.max]:
        sim = nearest[word]
        print(round(sim, 3), word)

