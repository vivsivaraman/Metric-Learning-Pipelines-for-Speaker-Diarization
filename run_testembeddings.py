#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 20:20:38 2019

@author: vivek


Extracts speaker embeddings for the CALLHOME. Use the embeddings extracted to perform diarization using run_diarization.py
(OR)
Extract speaker embeddings for the TEDLIUM dev dataset to visualize the embeddings. 
"""

from glob import glob
import h5py
import numpy as np
from os import path, makedirs
from sklearn.manifold import TSNE
import sys
import tensorflow as tf
from hyperparams import Hyperparams as hp
import utils
import Models.attention_inference


def main(args):
    with tf.Session() as sess:

        dataset = 'callhome'  # Which dataset to compute the embeddings, tedlium or callhome

        # Restore model. Get margin from params file in the chosen experiment folder.
        model = Models.attention_inference.Metriclearningmodel(loss_type=hp.loss_type, margin=hp.margin, sampling_type=hp.sampling_type, phase='test')

        #Should provide the path of the best validation step 
        model_path = path.join(hp.logdir, 'model_step_%06d' % 3200)
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        if dataset == 'callhome':

            data_dir = hp.callhome_data_dir
            feature_dir = path.join(data_dir, 'mfcc')
            emb_dir = path.join(data_dir, 'embs')
            if not path.exists(emb_dir):
                makedirs(emb_dir)

            # Get embeddings for each recording.
            for conv in glob(path.join(feature_dir, "*.hdf5")):
                rec_name = path.splitext(path.basename(conv))[0]

                with h5py.File(conv, 'r') as f:
                    features = f['mfcc'][:]  # Gives a numpy array

                # Use placeholder labels since they do not affect the embeddings (only used in calculating the loss).
                model.get_embeddings(sess, features, np.zeros((len(features, ))),
                                     save=True, save_path=path.join(emb_dir, rec_name + '.hdf5'))

        elif dataset == 'tedlium':

            data_dir = hp.tedlium_devdata_dir
            data, label = utils.load_data_ted_dev(data_dir)
            # Visualize for a subset of speakers.
            indices = label <= 19  # Select 20 speakers.
            data = data[indices]
            label = label[indices]
            embs, _ = model.get_embeddings(sess, data, label)

            embs_tsne = TSNE(n_components=2, init='pca', random_state=0).fit_transform(embs)
            utils.vis_tsne(embs_tsne, label)

        else:
            raise ValueError('Dataset not defined.')


if __name__ == '__main__':
    main(sys.argv)
