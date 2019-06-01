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
ap.add_argument("-s", "--similarity", type=str, default='jwxcos',
        help="Similarity: cos or jw or jwxcos")
ap.add_argument("-M", "--measure", type=str, default='average',
        help="Linkage measure average/complete/single")

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

def rellev(word, otherword):
    return 1 - editdistance.eval(word, otherword) / (len(word)+len(otherword))

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


logging.info('Read in root lemma forms')
root_lemma_forms = defaultdict(dict)
with open(args.roots_lemmas_forms) as fh:
    for line in fh:
        if args.lowercase:
            line = line.lower()
        words = line.split()
        root = words[0]
        lemma = words[1]
        forms = words[1:]  # the lemma is also a form
        root_lemma_forms[root][lemma] = set(forms)
logging.info('{} root lemma forms read'.format(len(root_lemma_forms)))

logging.info('Cluster forms')
all_true_labels = list()
all_predicted_labels = list()
all_forms = 0
all_correct = 0

if args.root:
    iterate_over = [args.root]
else:
    iterate_over = root_lemma_forms

for root in iterate_over:
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
    
    if args.baselines:
        pred_labels = labels.copy()
        random.shuffle(pred_labels)
        all_true_labels.extend(labels)
        all_predicted_labels.extend(pred_labels)
    else:
        # compute the distance matrix
        I = len(data)
        D = np.empty((I, I))
        dist_label_pairs = list()
        for i1 in range(I):
            for i2 in range(I):
                dist = get_dist(data[i1], data[i2])
                D[i1,i2] = dist
                if i1 < i2:
                    is_infl = labels[i1] == labels[i2]
                    dist_label_pairs.append( (dist, is_infl) )

        # for further analyses, sort the distances and separate into infls and noninfls
        dist_label_pairs.sort(key=lambda x: x[0])
        dist_infl = [dist for dist, is_infl in dist_label_pairs if is_infl]
        dist_ninf = [dist for dist, is_infl in dist_label_pairs if not is_infl]
        
        # print out
        if args.verbose:
            print()
            print("ROOT:", root)
            print("The cluster contains {} word forms belonging to {} lemmas: {}".format(
                len(data), cluster_num, " ".join(root_lemma_forms[root].keys()) ))

        if args.analthresh or args.prfplot:
            total = 0
            infl = 0
            infl_total = len(dist_infl)
            dprf = list()
            for dist, is_infl in dist_label_pairs:
                total += 1
                if is_infl:
                    infl += 1
                prec = infl/total
                rec = infl/infl_total
                if prec + rec > 0:
                    f = 2 * prec * rec / (prec + rec)
                else:
                    f = 0
                dprf.append( (dist, prec, rec, f) )
        
        if args.analthresh:
            # analyze optimal threshold
            best_f = -1
            best_thresh = -1
            for dist, _, _, f in prf:
                if f > best_f:
                    best_f = f
                    best_thresh = dist
            if args.verbose:
                print("BEST THRESH:", best_thresh, "with f1:", best_f)
                print("ALLINFL:", *dist_infl)
                print("ALLNOIN:", *dist_ninf)
            else:
                print(best_thresh, best_f)

        elif args.plot:
            n_bins = I
            fig, ax = plt.subplots(figsize=(8, 4))
            n, bins, patches = ax.hist(dist_infl, n_bins, density=True, histtype='step',
                           cumulative=True, label='Infl')
            plot_dist_ninf = dist_ninf[:len(dist_infl)]
            ax.hist(plot_dist_ninf, bins=bins, density=True, histtype='step',
                    cumulative=True, label='Noninfl')
            ax.grid(True)
            ax.legend(loc='right')
            ax.set_title('Empirical CDF for forms of {}'.format(root))
            ax.set_xlabel('Word form distance threshold, ' + args.similarity)
            ax.set_ylabel('Proportion of word form pairs, CDF')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.savefig(args.plot+"-"+root+".png")
        
        elif args.prfplot:
            dists = [x[0] for x in dprf]
            precs = [x[1] for x in dprf]
            recs  = [x[2] for x in dprf]
            fs    = [x[3] for x in dprf]

            plt.plot(dists, precs, label='precision')
            plt.plot(dists, recs, label='recall')
            plt.plot(dists, fs, label='F1')
            plt.grid(True)
            plt.legend()
            plt.title('Inflection identification for forms of {}'.format(root))
            plt.xlabel('Word form distance threshold, ' + args.similarity)
            plt.ylabel('Inflection pairs below threshold')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.savefig(args.prfplot+"-"+root+".png")
        
        else:
            # cluster the forms
            clustering = AgglomerativeClustering(affinity='precomputed',
                    linkage = args.measure, n_clusters=cluster_num)
            clustering.fit(D)
            
            if args.verbose:
                cluster_elements = defaultdict(set)
                for word, lemma, cluster in zip(data, labels, clustering.labels_):
                    cluster_elements[cluster].add((word, lemma))
                for cluster in sorted(cluster_elements):
                    contents = ["{} [{}],".format(word, lemma)
                            for word, lemma in cluster_elements[cluster]]
                    print(cluster, *contents)

            # eval
            word2cluster = {word: cluster for word, cluster in zip(data, clustering.labels_)}
            correct = sum([1 for word, lemma in zip(data, labels)
                if word != lemma and word2cluster[word] == word2cluster[lemma]])
            # total count of forms = all forms - lemmas
            total = I - cluster_num
            # ratio of forms in the same cluster as their lemmas
            lemmaacc = correct / total

            #hcv = homogeneity_completeness_v_measure(labels, clustering.labels_)
            
            # logging.info("HCV: " + " ".join([str(x) for x in hcv]))
            #print("HCV:", *hcv, flush=True)
            print(lemmaacc, *hcv, flush=True)
            
            all_forms += total
            all_correct += correct
            #all_true_labels.extend(labels)
            #all_predicted_labels.extend([root+str(label) for label in clustering.labels_])


#if all_predicted_labels and not args.root:
#    hcv = homogeneity_completeness_v_measure(all_true_labels,
#            all_predicted_labels)
#    lemmaacc = all_correct / all_forms
#    print(lemmaacc, *hcv)

