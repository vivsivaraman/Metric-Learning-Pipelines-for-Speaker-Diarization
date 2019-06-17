
"""Perform the diarization using the embeddings from the trained network."""

import csv
from glob import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
from os import path
from pyannote.core import notebook
import sys
from hyperparams import Hyperparams as hp

from Models.diarization_aux_funcs import perform_clustering, get_annotations, get_der


def main(args):

    emb_dir = hp.callhome_emb_dir
    label_dir = hp.callhome_label_dir
    pred_dir = hp.callhome_pred_dir

    # Clustering per conversation.
    cluster_method = 'kmeans'
    save_dir = hp.logdir
    save_file = 'der_rates_v1.csv'

    der_rates = []
    aver_der_rate = 0

    do_plot = False  # Whether to check the annotation plot for target.
    target = '4822'  #Visualize the plots for this target

    # Perform diarization for each conversation.
    for conv in glob(path.join(emb_dir, "*.hdf5")):

        rec_name = path.splitext(path.basename(conv))[0]

        with h5py.File(conv, 'r') as f:
            embs = f['embs'][:]  # Gives a numpy array
        # Perform PCA.
        # embs = pca.fit_transform(embs)
        if np.isnan(embs).any() or np.isinf(embs).any():
            raise ValueError('Embeddings contain invalid value.')

        pred_labels = perform_clustering(embs, method=cluster_method)

        true_annotation, pred_annotation = get_annotations(rec_name, pred_labels, label_dir, pred_dir)

        if do_plot:

            if rec_name != target:  # Loop to find the target without calculating DER.
                continue
            notebook.width = 40
            plt.rcParams['figure.figsize'] = (notebook.width, 5)
            # plot reference
            plt.subplot(211)
            notebook.plot_annotation(true_annotation, legend=True, time=False)
            plt.ylabel('Reference', fontsize=16)
            # plot hypothesis
            plt.subplot(212)
            notebook.plot_annotation(pred_annotation, legend=True, time=True)
            plt.ylabel('Hypothesis', fontsize=16)
            plt.show()

            return

        der_rate = get_der(true_annotation, pred_annotation)
        der_rates.append([rec_name, der_rate])
        aver_der_rate += der_rate

    print(aver_der_rate / float(len(der_rates)))
    with open(path.join(save_dir, save_file), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(der_rates)


if __name__ == '__main__':
    main(sys.argv)
