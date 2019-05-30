#!/usr/bin/env python3
#coding: utf-8

from collections import defaultdict
import sys
import logging
import argparse

from pyjarowinkler import distance
import unidecode
import fasttext

#from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy import inner
from numpy.linalg import norm
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.cluster import AgglomerativeClustering

ap = argparse.ArgumentParser(
        description='cluster forms of all lemmas in a dertree')
ap.add_argument('embeddings',
        help='file with the embeddings')
ap.add_argument('roots_lemmas_forms',
        help='file with the dertree roots and forms and lemmas')
ap.add_argument("-s", "--similarity", type=str, default='jwxcos',
        help="Similarity: cos or jw or jwxcos")
ap.add_argument("-M", "--measure", type=str, default='average',
        help="Linkage measure average/complete/single")
ap.add_argument("-V", "--verbose", action="store_true",
        help="Print more verbose progress info")
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
    uword = devow(word)
    uotherword = devow(otherword)
    usim = jw_safe(uword, uotherword)    
    sim = (sim+usim)/2
    assert sim >= 0 and sim <= 1, "JW sim must be between 0 and 1"
    return sim

def lensim(word, otherword):
    return 1 / (1 + args.length * abs(len(word) - len(otherword)) )

def similarity(word, otherword):
    if args.similarity == 'jw':
        return jwsim(word, otherword)
    elif args.similarity == 'jwxcos':
        return jwsim(word, otherword) * embsim(word, otherword)
    elif args.similarity == 'jwxcosxlen':
        return jwsim(word, otherword) * embsim(word, otherword) * lensim(word, otherword);
    elif args.similarity == 'len':
        return lensim(word, otherword);
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
            embedding[word] = embedding
logging.info('Embeddings read')


logging.info('Read in root lemma forms')
root_lemma_forms = defaultdict(dict)
with open(args.roots_lemmas_forms) as fh:
    for line in fh:
        words = line.split()
        root = words[0]
        lemma = words[1]
        forms = words[1:]  # the lemma is also a form
        root_lemma_forms[root][lemma] = forms
logging.info('{} root lemma forms read'.format(len(root_lemma_forms)))

logging.info('Cluster forms')
all_true_labels = list()
all_predicted_labels = list()
for root in root_lemma_forms:
    logging.info("Clustering forms for dertree '{}'".format(root))
    data = list()  # the forms
    labels = list()  # their lemmas
    for lemma in root_lemma_forms[root]:
        # cluster all the forms of all the lemmas
        forms = root_lemma_forms[root][lemma]
        data.extend(forms)
        # the lemmas are the cluter labels
        lemmas = [lemma for form in forms]
        labels.extend(lemmas)
    # number of lemmas is number of clusters
    cluster_num = len(root_lemma_forms[root])
    # cluster the forms
    I = len(data)
    D = np.empty((I, I))
    for i1 in range(I):
        for i2 in range(I):
            D[i1,i2] = get_dist(data[i1], data[i2])
    clustering = AgglomerativeClustering(affinity='precomputed',
            linkage = args.measure, n_clusters=cluster_num)
    clustering.fit(D)
    
    # print out
    print("ROOT:", root)
    cluster_elements = defaultdict(set)
    for word, lemma, cluster in zip(data, labels, clustering.labels_):
        cluster_elements[cluster].add((word, lemma))
    for cluster in cluster_elements:
        contents = ["{} [{}],".format(word, lemma)
                for word, lemma in cluster_elements[cluster]]
        print(cluster, *contents)

    # eval
    hcv = homogeneity_completeness_v_measure(labels, clustering.labels_)
    logging.info(" ".join(hcv))
    all_true_labels.extend(labels)
    all_predicted_labels.extend([root+str(label) for label in clustering.labels_])

hcv = homogeneity_completeness_v_measure(all_true_labels,
        all_predicted_labels)
logging.info("Total HCV:")
logging.info(" ".join(hcv))

