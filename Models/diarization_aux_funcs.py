#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 19:22:52 2019

@author: vivek
"""

import numpy as np
from os import path
from pyannote.core import Annotation, Segment, json
from pyannote.metrics.diarization import DiarizationErrorRate
from sklearn.cluster import KMeans, SpectralClustering
#from pyrcc import rcc
from hyperparams import Hyperparams as hp
#from ivec_kmeans import IvecKMeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import xmeans
from sklearn.metrics.pairwise import cosine_similarity


def get_annotations(rec_name, pred_labels, label_dir, pred_dir):

    true_annotation = json.load_from(path.join(label_dir, rec_name + '.json'))
    # Modify region extent from sample index to time.
    for seg in true_annotation.itersegments():
        new_seg = Segment(seg.start / hp.callhome_rate, seg.end / hp.callhome_rate)
        seg_label = true_annotation[seg]
        del true_annotation[seg]
        true_annotation[new_seg] = seg_label

    # Create prediction annotation.
    starts_ends = np.loadtxt(path.join(pred_dir, rec_name + '.csv'), dtype=int, delimiter=',', usecols=[1, 2])
    pred_annotation = Annotation()
    for i, start_end in enumerate(starts_ends):
        cur_seg = Segment(start_end[0] / hp.callhome_rate, start_end[1] / hp.callhome_rate)
        pred_annotation[cur_seg] = pred_labels[i]

    return true_annotation, pred_annotation


def est_num_clusters(embs, max_num, init_num):
    """Use xmeans to estimate number of speakers."""

    embs_list = embs.tolist()
    initial_centers = kmeans_plusplus_initializer(embs_list, init_num).initialize()
    xm = xmeans(embs_list, initial_centers, kmax=max_num, ccore=True)
    xm.process()
    num_speakers = len(xm.get_clusters())
    print('Estimated number of speakers: ' + str(num_speakers))

    return num_speakers


def perform_clustering(embs, method='kmeans'):
    """Perform clusering based on specified method."""

    if method == 'kmeans':

        num_speakers = est_num_clusters(embs, max_num=7, init_num=2)
        cluster = KMeans(n_clusters=num_speakers, init='k-means++', n_jobs=-1, random_state=0)
        # cluster.fit_predict(embs)
        # pred_labels = cluster.labels_
        cluster.fit(embs)
        centroids = cluster.cluster_centers_
        pred_labels = np.argmax(cosine_similarity(embs, centroids), axis=1)

    elif method == 'plda_kmeans':

        num_speakers = est_num_clusters(embs, max_num=7, init_num=2)
        cluster = IvecKMeans(np.array(kmeans_plusplus_initializer(embs, num_speakers).initialize()),
                             num_speakers, score_method='plda')
        cluster.fit(embs)
        pred_labels = cluster.old_labels

    elif method == 'cosine_kmeans':

        num_speakers = est_num_clusters(embs, max_num=7, init_num=2)
        cluster = IvecKMeans(np.array(kmeans_plusplus_initializer(embs, num_speakers).initialize()),
                             num_speakers, score_method='cosine')
        cluster.fit(embs)
        pred_labels = cluster.labels()

    elif method == 'rcc':
        cluster = rcc.RccCluster(k=10, measure='cosine', clustering_threshold=1, verbose=False)
        pred_labels = cluster.fit(embs)

    elif method == 'spectral':
        cluster = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=10, n_jobs=-1)
        cluster.fit_predict(embs)
        pred_labels = cluster.labels_

    elif method == 'spectral_cosine':
        sigma_squared = 0.5
        cosine_dist = 1 - cosine_similarity(embs)
        affinity = np.exp(-np.power(cosine_dist, 2) / sigma_squared)
        if np.isnan(affinity).any() or np.isinf(affinity).any():
            raise ValueError('Affinity matrix contains NaN.')
        norms = np.linalg.norm(embs, axis=1)
        print(np.max(norms), np.min(norms))
        print(np.max(affinity), np.min(affinity))
        cluster = SpectralClustering(n_clusters=2, affinity='precomputed', n_jobs=-1)
        cluster.fit(affinity)
        pred_labels = cluster.labels_

    else:
        raise ValueError('Clustering method not defined.')

    return pred_labels


def get_der(true_annotation, pred_annotation):
    """Calculate Diarization Error Rate - only the confusion. """

    metric = DiarizationErrorRate(collar=0.5)
    start = true_annotation.get_timeline().extent().start
    end = true_annotation.get_timeline().extent().end
    components = metric(true_annotation, pred_annotation, detailed=True, uem=Segment(start, end))
    der_rate = components['confusion'] / components['total']  # Only consider confusion.
    print("DER = {0:.3f}".format(der_rate))

    return der_rate
