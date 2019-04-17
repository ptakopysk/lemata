#!/usr/bin/env python3
#coding: utf-8



from czech_stemmer import cz_stem

import argparse
import sys
from collections import defaultdict, Counter
from sortedcollections import ValueSortedDict
from collections import OrderedDict

from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import homogeneity_completeness_v_measure

from numpy import inner
from numpy.linalg import norm

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from ast import literal_eval as make_tuple
from sklearn.metrics import confusion_matrix
import itertools

from sklearn.cluster import AgglomerativeClustering

from pyjarowinkler import distance

import unidecode

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging

ap = argparse.ArgumentParser(
        description='find lemma for form as nearest lemma in emb space')
ap.add_argument('embeddings',
        help='file with the embeddings')
ap.add_argument('conllu_test',
        help='file with the forms and lemmas')

ap.add_argument("-l", "--lowercase", action="store_true",
        help="lowercase input forms")
ap.add_argument("-S", "--stems", type=int, default=2,
        help="Use stems of length S (first S characters, but see also M and D)")
ap.add_argument("-R", "--remerge", type=int,
        help="Remerge clusters on test data using stem length R")
ap.add_argument("-r", "--remergethreshold", type=float, default=0.3,
        help="threshold when remerging")
ap.add_argument("-D", "--devow", action="store_true",
        help="Devowel stems")
ap.add_argument("-P", "--postags", type=str,
        help="Read in a POS tag disctionary and add POS to stems")

ap.add_argument("-n", "--number", type=int,
        help="How many embeddings to read in")
ap.add_argument("-V", "--verbose", action="store_true",
        help="Print more verbose progress info")
ap.add_argument("-N", "--normalize", action="store_true",
        help="Normalize the embeddings")
ap.add_argument("-b", "--baselines", action="store_true",
        help="Compute baselines and upper bounds")
ap.add_argument("-t", "--threshold", type=float, default=0.30,
        help="Do not perform merges with avg distance greater than this")
ap.add_argument("-O", "--oov", type=str, default="guess",
        help="OOVs: keep/guess")
# TODO unued
ap.add_argument("-p", "--plot", type=str,
        help="Plot the dendrogramme for the given stem")
ap.add_argument("-m", "--merges", action="store_true",
        help="Write out the merges")
ap.add_argument("-s", "--similarity", type=str,
        help="Similarity: cos or jw")
ap.add_argument("-C", "--clusters", action="store_true",
        help="Print out the clusters.")
ap.add_argument("-L", "--length", type=float, default=0.05,
        help="Weight for length similarity")
args = ap.parse_args()



level = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=level)


# TODO how to do this right?
OOV_EMB_SIM = 0.9


# https://github.com/scikit-learn/scikit-learn/blob/70cf4a676caa2d2dad2e3f6e4478d64bcb0506f7/examples/cluster/plot_hierarchical_clustering_dendrogram.py
# Authors: Mathew Kallada
# License: BSD 3 clause
"""
=========================================
Plot Hierarachical Clustering Dendrogram 
=========================================
This example plots the corresponding dendrogram of a hierarchical clustering
using AgglomerativeClustering and the dendrogram method available in scipy.
"""

import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

def plot_dendrogram(model, **kwargs):


    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

    plt.xticks(rotation=90)
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.2)




# unidecode and remove vowels
def devow(form):
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
    return sim

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
    uword = unidecode.unidecode(word)
    uotherword = unidecode.unidecode(otherword)
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

if args.postags:
    logging.info('Read in POS tag dictionary')
    # TODO save most frequent tag (now last occurring tag)
    postag = defaultdict(lambda: 'NOUN')
    with open(args.postags) as conllufile:
        for line in conllufile:
            fields = line.split()
            if fields and fields[0].isdecimal():
                assert len(fields) > 2
                form = fields[1]
                pos = fields[2]
                if args.lowercase:
                    form = form.lower()
                postag[form] = pos

def get_stem(form, remerging=False):
    if args.lowercase:
        form = form.lower()

    if args.devow:
        form = devow(form)

    if remerging:
        stem = form[:args.remerge]
    else:
        stem = form[:args.stems]
    
    if args.postags:
        stem = stem + '_' + postag[form]

    return stem
    # return cz_stem(form, aggressive=False)

logging.info('Read in embeddings')
embedding = defaultdict(list)
forms_stemmed = defaultdict(set)
form_freq_rank = dict()
with open(args.embeddings) as embfile:
    size, dim = map(int, embfile.readline().split())
    if args.number:
        size = min(size, args.number)
    for i in range(size):
        fields = embfile.readline().split()
        form = fields[0]
        emb = list(map(float, fields[1:]))
        if args.normalize:
            emb /= norm(emb)
        if args.lowercase and not form.islower():
            form = form.lower()
            if form in embedding:
                # do not overwrite "bush" with "Bush"
                continue
        embedding[form] = emb
        stem = get_stem(form)
        forms_stemmed[stem].add(form)
        form_freq_rank[form] = i

if args.verbose:
    for form in sorted(embedding.keys()):
        logging.debug(form + ' -> ' + get_stem(form))

logging.info('Read in test form-lemma pairs')
test_data = list()
with open(args.conllu_test) as conllufile:
    for line in conllufile:
        fields = line.split()
        if fields and fields[0].isdecimal():
            assert len(fields) > 2
            form = fields[1]
            lemma = fields[2]
            # pos = fields[3]
            if args.lowercase:
                form = form.lower()
                lemma = lemma.lower()
            test_data.append((form, lemma))
logging.info('Done reading')

def get_dist(form1, form2):
    # similarity to distance
    return 1-similarity(form1, form2)


# list of indexes -> list of words
def node2str(node, index2word):
    return [index2word[index] for index in node]

def linkage(cluster1, cluster2, D):
    linkages = list()
    for node1 in cluster1:
        for node2 in cluster2:
            linkages.append(D[node1, node2])
    # min avg max
    # return min(linkages), sum(linkages)/len(linkages), max(linkages)
    # avg
    return sum(linkages)/len(linkages)


# cluster each hypercluster
logging.info('Run the main loop')

#iterate_over = forms_stemmed
#if args.plot:
#    iterate_over = [args.plot]

def cl(stem, cluster):
    return stem + '___' + str(cluster)

def aggclust(forms_stemmed):
    # form -> cluster
    result = dict()
    for stem in forms_stemmed:
        # vocabulary
        index2word = list(forms_stemmed[stem])
        I = len(index2word)
        
        logging.debug(stem)
        logging.debug(I)
        logging.debug(index2word)
        
        if I == 1:
            result[index2word[0]] = cl(stem, 0)
            continue

        D = np.empty((I, I))
        for i1 in range(I):
            for i2 in range(I):
                D[i1,i2] = get_dist(index2word[i1], index2word[i2])
        clustering = AgglomerativeClustering(affinity='precomputed',
                linkage = 'average', n_clusters=1)
        clustering.fit(D)

        # default: each has own cluster
        clusters = list(range(I))
        nodes = [[i] for i in range(I)]
        for merge in clustering.children_:
            # check stopping criterion
            if args.threshold < linkage(nodes[merge[0]], nodes[merge[1]], D):
                break
            # perform the merge
            nodes.append(nodes[merge[0]] + nodes[merge[1]])
            # reassign words to new cluster ID
            for i in nodes[-1]:
                clusters[i] = len(nodes) - 1
        for i, cluster in enumerate(clusters):
            result[index2word[i]] = cl(stem, cluster)
    return result
                

#if args.plot:
#        plt.title('Hierarchical Clustering Dendrogram')
#        plot_dendrogram(clustering, labels=index2word)
#        plt.show()

def writeout_clusters(clustering):
    cluster2forms = defaultdict(list)
    for form, cluster in clustering.items():
        cluster2forms[cluster].append(form)
    for cluster in sorted(cluster2forms.keys()):
        print('CLUSTER', cluster)
        for form in cluster2forms[cluster]:
            print(form)
        print()
    sys.stdout.flush()

clusterset = set()

# each cluster name becomes its most frequent wordform
def rename_clusters(clustering):
    cluster2forms = defaultdict(list)
    for form, cluster in clustering.items():
        cluster2forms[cluster].append(form)

    cluster2newname = dict()
    for cluster, forms in cluster2forms.items():
        form2rank = dict()
        for form in forms:
            assert form in form_freq_rank
            form2rank[form] = form_freq_rank[form]
        most_frequent_form = min(form2rank, key=form2rank.get)
        cluster2newname[cluster] = most_frequent_form
        clusterset.add(most_frequent_form)

    new_clustering = dict()
    for form, cluster in clustering.items():
        new_clustering[form] = cluster2newname[cluster]

    return new_clustering

# now 1 nearest neighbour wordform;
# other option is nearest cluster in avg linkage
# (probably similar result but not necesarily)
def find_cluster_for_form(form, clustering):
    stem = get_stem(form)
    cluster = form  # backoff: new cluster
    if args.oov == "guess" and stem in forms_stemmed:
        dists = dict()
        for otherform in forms_stemmed[stem]:
            dists[otherform] = get_dist(form, otherform)
        nearest_form = min(dists, key=dists.get)
        if dists[nearest_form] < args.threshold:
            cluster = clustering[nearest_form]
            # else leave the default, i.e. a separate new cluster
    return cluster


clusters_restemmed = defaultdict(list)
cluster_remerged = dict()

def remerge(pivot_cluster):
    merged_clusters = set()
    merged_clusters.add(pivot_cluster)
    stem = get_stem(pivot_cluster, remerging=True)
    for candidate_cluster in clusters_restemmed[stem]:
        # find all near clusters; just look at the representant words
        if get_dist(pivot_cluster, candidate_cluster) < args.remergethreshold:
            merged_clusters.add(candidate_cluster)
    # find name for the new merghed cluster
    form2rank = dict()
    form2rank[pivot_cluster] = args.number
    for form in merged_clusters:
        if form in form_freq_rank:
            form2rank[form] = form_freq_rank[form]
    merged_name = min(form2rank, key=form2rank.get)
    # define the merge
    print('MERGE:', merged_name, merged_clusters)
    for cluster in merged_clusters:
        cluster_remerged[cluster] = merged_name


def homogeneity(clustering, writeout=False):
    golden = list()
    predictions = list()
    lemmatization_corrects = 0
    found_clusters = dict()  # caching
    lemma2clusters2forms = defaultdict(lambda: defaultdict(set))
    if args.remerge:
        for cluster in clusterset:
            clusters_restemmed[get_stem(cluster, remerging=True)].append(cluster)
    
    for form, lemma in test_data:
        golden.append(lemma)
        if form in clustering:
            cluster = clustering[form]
        else:
            if form not in found_clusters:
                found_clusters[form] = find_cluster_for_form(form, clustering)
            cluster = found_clusters[form]
        if lemma in clustering:
            lemmacluster = clustering[lemma]
        else:
            if lemma not in found_clusters:
                found_clusters[lemma] = find_cluster_for_form(lemma, clustering)
            lemmacluster = found_clusters[lemma]

        if args.remerge:
            if cluster not in cluster_remerged:
                remerge(cluster)
            cluster = cluster_remerged[cluster]
            if lemmacluster not in cluster_remerged:
                remerge(lemmacluster)
            lemmacluster = cluster_remerged[lemmacluster]

        predictions.append(cluster)
        lemma2clusters2forms[lemma][cluster].add(form)
        if cluster == lemmacluster:
            lemmatization_corrects += 1
        if writeout:
            oov = 'OOVform' if form in found_clusters else ''
            lemmaoov = 'OOVlemma' if lemma in found_clusters else ''
            dist = get_dist(form, lemma)
            good = 'GOOD' if cluster == lemmacluster else 'BAD'
            print(oov, form, '->', cluster, good,
                    '{:.4f}'.format(dist), lemmaoov, lemma, '->', lemmacluster)
    if writeout:
        print('PER LEMMA WRITEOUT')
        for lemma in lemma2clusters2forms:
            print('LEMMA:', lemma)
            for cluster in lemma2clusters2forms[lemma]:
                print(get_stem(cluster), cluster, ':', lemma2clusters2forms[lemma][cluster])
            print()
    return (homogeneity_completeness_v_measure(golden, predictions), lemmatization_corrects/len(golden))

def baseline_clustering(test_data, basetype):
    result = dict()
    for form, lemma in test_data:
        for word in (form, lemma):
            stem = get_stem(word)
            if basetype == 'formlemma':
                result[word] = cl(stem, word)
            elif basetype == 'stemlemma':
                result[word] = cl(stem, 0)
            elif basetype == 'upper':
                result[word] = cl(stem, lemma)
            elif basetype == 'stem5':
                result[word] = cl(stem, word[:5])
            logging.debug(basetype + ': ' + word + ' -> ' + result[word])
    return result


if args.baselines:
    logging.info('Run evaluation')
    known = 0
    unknown = 0
    for form, _ in test_data:
        if form in embedding:
            known += 1
        else:
            unknown += 1
    print('OOV rate:', unknown, '/', (known+unknown), '=',
            (unknown/(known+unknown)*100))

    print('Type', 'homogeneity', 'completenss', 'vmeasure', 'accuracy', sep='\t')
    for basetype in ('formlemma', 'stemlemma', 'stem5', 'upper'):
        clustering = baseline_clustering(test_data, basetype)
        hcva = homogeneity(clustering)
        print(basetype, *hcva, sep='\t')
else:
    clustering = aggclust(forms_stemmed)
    logging.info('Rename clusters')
    renamed_clustering = rename_clusters(clustering)
    if args.clusters:
        logging.info('Write out train clusters')
        print('START TRAIN CLUSTERS')
        writeout_clusters(renamed_clustering)
        print('END TRAIN CLUSTERS')
    logging.info('Run evaluation')
    hcv = homogeneity(renamed_clustering, writeout=args.clusters)
    print('Homogeneity', 'completenss', 'vmeasure', 'accuracy', sep='\t')
    print(*hcva, sep='\t')

logging.info('Done.')

