#!/usr/bin/env python3
#coding: utf-8

from collections import defaultdict
import sys
import logging
import argparse
import random

from pyjarowinkler import distance
import unidecode
import fasttext
import editdistance

#from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy import inner
from numpy.linalg import norm
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.cluster import AgglomerativeClustering

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(
        description='cluster forms of all lemmas in a dertree')
ap.add_argument('embeddings',
        help='file with the embeddings')
ap.add_argument('roots_lemmas_forms',
        help='file with the dertree roots and forms and lemmas')

ap.add_argument("-l", "--lowercase", action="store_true",
        help="lowercase input forms")
ap.add_argument("-S", "--simplify", action="store_true",
        help="simplify input forms for edit dost")
ap.add_argument("-s", "--similarity", type=str, default='jwxcos',
        help="Similarity: cos or jw or jwxcos")
ap.add_argument("-M", "--measure", type=str, default='average',
        help="Linkage measure average/complete/single")

ap.add_argument("-E", "--eval", type=str, default="lemmatization",
        help="what to evaluate -- lemmatization or pairs")
ap.add_argument("-R", "--root", type=str,
        help="Only process the given root")
ap.add_argument("-p", "--plot", type=str,
        help="Plot the CDF under the given file prefix")
ap.add_argument("-P", "--prfplot", type=str,
        help="Plot the prec rec and F under the given file prefix")
ap.add_argument("-T", "--analthresh", action="store_true",
        help="Analyze the best threshold")
ap.add_argument("-b", "--baselines", action="store_true",
        help="Compute baselines and upper bounds")
ap.add_argument("-V", "--verbose", action="store_true",
        help="Print more verbose progress info")
ap.add_argument("-O", "--output", action="store_true",
        help="Print results to standard output")
args = ap.parse_args()

level = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=level)

OOV_EMB_SIM = 0.9


# @functools.lru_cache(maxsize=1000000)
def devow(form):
    """unidecode and remove non-initial vowels"""

    # implicit transliteration and deaccentization
    uform = unidecode.unidecode(form)

    # keep first letter
    dform = uform[1:]
    # remove vowels, do not presuppose lowercasing
    dform = dform.replace("a", "")
    dform = dform.replace("e", "")
    dform = dform.replace("i", "")
    dform = dform.replace("o", "")
    dform = dform.replace("u", "")
    dform = dform.replace("y", "")
    dform = dform.replace("A", "")
    dform = dform.replace("E", "")
    dform = dform.replace("I", "")
    dform = dform.replace("O", "")
    dform = dform.replace("U", "")
    dform = dform.replace("Y", "")
    
    return uform[:1] + dform


def embsim(word, otherword):
    """Cosine similarity of wrod embeddings.

    Shifted into the 0..1 range."""

    if word in embedding and otherword in embedding:
        emb1 = embedding[word]
        emb2 = embedding[otherword]
        sim = inner(emb1, emb2)/(norm(emb1)*norm(emb2))
        # sim = cosine_similarity([emb1], [emb2])
        #logging.debug(sim)
        assert sim >= -1.0001 and sim <= 1.0001, "Cos sim must be between -1 and 1"
        # shift to 0..1 range
        sim = (sim+1)/2
    else:
        # backoff
        sim = OOV_EMB_SIM
        logging.warning("Embedding unknown for '{}', '{}'".format(word, otherword))
    return sim

def jw_safe(srcword, tgtword):
    """Jaro Winkler similarity that can take emtpy words.

    Is in 0..1 range.
    """

    if srcword == '' or tgtword == '':
        # 1 if both empty
        # 0.5 if one is length 1
        # 0.33 if one is length 2
        # ...
        return 1/(len(srcword)+len(tgtword)+1)
    elif srcword == tgtword:
        return 1
    else:
        # called distance but is actually similarity
        return distance.get_jaro_distance(srcword, tgtword)

def jwsim(word, otherword):
    sim = jw_safe(word, otherword)
    if args.simplify:
        uword = devow(word)
        uotherword = devow(otherword)
        usim = jw_safe(uword, uotherword)    
        sim = (sim+usim)/2
    assert sim >= 0 and sim <= 1, "JW sim must be between 0 and 1"
    return sim

def _rellev(word, otherword):
    return 1 - editdistance.eval(word, otherword) / (len(word)+len(otherword))

def rellev(word, otherword):
    sim = _rellev(word, otherword)
    if args.simplify:
        uword = devow(word)
        uotherword = devow(otherword)
        usim = _rellev(uword, uotherword)    
        sim = (sim+usim)/2
    return sim

def lensim(word, otherword):
    return 1 / (1 + args.length * abs(len(word) - len(otherword)) )

def similarity(word, otherword):
    if args.lowercase:
        word = word.lower()
        otherword = otherword.lower()
    
    if args.similarity == 'jw':
        return jwsim(word, otherword)
    elif args.similarity == 'jwxcos':
        return jwsim(word, otherword) * embsim(word, otherword)
    elif args.similarity == 'jwxcosxlen':
        return jwsim(word, otherword) * embsim(word, otherword) * lensim(word, otherword);
    elif args.similarity == 'len':
        return lensim(word, otherword);
    elif args.similarity == 'lev':
        return rellev(word, otherword);
    else:
        # cos
        return embsim(word, otherword)

def get_dist(form1, form2):
    # similarity to distance
    return 1-similarity(form1, form2)

#def linkage(cluster1, cluster2, D):
#    linkages = list()
#    for node1 in cluster1:
#        for node2 in cluster2:
#            linkages.append(D[node1, node2])
#    # min avg max
#    if args.measure == 'average':
#        return sum(linkages)/len(linkages)
#    elif args.measure == 'single':
#        return min(linkages)
#    elif args.measure == 'complete':
#        return max(linkages)
#    else:
#        assert False




def eval_clust(data, labels, pred_labels, cluster_num, I):
    cluster_elements = defaultdict(set)
    for word, lemma, cluster in zip(data, labels, pred_labels):
        cluster_elements[cluster].add((word, lemma))
    if args.verbose:
        for cluster in sorted(cluster_elements):
            contents = ["{} [{}],".format(word, lemma)
                    for word, lemma in cluster_elements[cluster]]
            print(cluster, *contents)
    if args.output:
        for cluster in sorted(cluster_elements):
            contents = ["[{}] {}".format(lemma, word)
                    for word, lemma in cluster_elements[cluster]]
            print("\nCLUSTER {}:".format(cluster), *sorted(contents), sep="\n")
        print()

        print('PERLEMMA')
        lemma2formclusters = defaultdict(list)
        for cluster in cluster_elements:
            for form, lemma in cluster_elements[cluster]:
                lemma2formclusters[lemma].append((cluster, form))
        for lemma in lemma2formclusters:
            print()
            print('LEMMA {}:'.format(lemma))
            lemma2formclusters[lemma].sort(key=lambda x: x[0])
            for cluster, form in lemma2formclusters[lemma]:
                print(cluster, form, sep="\t")


    # eval
    word2cluster = {word: cluster for word, cluster in zip(data, pred_labels)}
    
    if args.eval == "lemmatization":
        # form is in same cluster as its lemma
        correct = sum([1 for word, lemma in zip(data, labels)
            if word != lemma and word2cluster[word] == word2cluster[lemma]])
        
        # total count of forms = all forms - lemmas
        total_forms = I - cluster_num
        rec = correct / total_forms
        
        # total sizes of clusters with lemmas
        # (clusters with multiple lemmas counted multiple times,
        # clusters without lemmas not counted)
        total_cluster_sizes = sum([len(cluster_elements[cluster])
            for word, lemma, cluster
            in zip(data, labels, pred_labels)
            if word == lemma])
        prec = correct / total_cluster_sizes
    elif args.eval == "pairs":
        correct = 0
        noninfl_in_cluster = 0
        infl_outside_cluster = 0
        for i1 in range(I):
            for i2 in range(i1+1, I):
                is_infl = labels[i1] == labels[i2]
                is_clust = pred_labels[i1] == pred_labels[i2]
                if is_infl and is_clust:
                    correct += 1
                elif is_infl and not is_clust:
                    infl_outside_cluster += 1
                elif not is_infl and is_clust:
                    noninfl_in_cluster += 1
        prec = correct / (correct + noninfl_in_cluster)
        rec = correct / (correct + infl_outside_cluster)
    else:
        assert False

    if prec + rec > 0:
        f = 2*prec*rec/(prec+rec)
    else:
        f = 0

    print(prec, rec, f, flush=True)




logging.info('Read in embeddings')
embedding = dict()
if args.embeddings.endswith('.bin'):
    # get word embedding still the same way, i.e. as embedding[word]
    # (iterate over embedding.words if at all needed)
    embedding = fasttext.load_model(args.embeddings)
else:
    with open(args.embeddings) as fh:
        for line in fh:
            fields = line.split()
            word = fields[0]
            vector = [float(x) for x in fields[1:]]
            embedding[word] = vector
logging.info('Embeddings read')


logging.info('Read in pairs and compute their dostacnes')
with open(args.roots_lemmas_forms) as fh:
    for line in fh:
        if args.lowercase:
            line = line.lower()
        words = line.rstrip('\n').split('\t')
        inflderi = words[0]
        form1 = words[1]
        tag1 = words[2]
        count1 = words[3]
        form2 = words[4]
        tag2 = words[5]
        count2 = words[6]

        dist = get_dist(form1, form2)

        print(count1, count2, inflderi, dist, form1, form2, tag1, tag2,
                sep="\t")

logging.info('Done')
