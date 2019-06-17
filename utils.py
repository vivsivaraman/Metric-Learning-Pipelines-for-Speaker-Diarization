"""Utility functions."""

import h5py
import matplotlib.pyplot as plt
import numpy as np
from os import path
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.utils.linear_assignment_ import linear_assignment
from vbs_demo.python import features
import codecs
import unicodedata
import tensorflow as tf

from hyperparams import Hyperparams as hp


class STMSegment(object):
    r"""
    For TED processing: Representation of an individual segment in an STM file.
    """

    def __init__(self, stm_line):
        tokens = stm_line.split()
        self._filename = tokens[0]
        self._channel = tokens[1]
        self._speaker_id = tokens[2]
        self._start_time = float(tokens[3])
        self._stop_time = float(tokens[4])
        self._labels = tokens[5]
        self._transcript = ""
        for token in tokens[6:]:
            self._transcript += token + " "
        # We need to do the encode-decode dance here because encode
        # returns a bytes() object on Python 3, and text_to_char_array
        # expects a string.
        self._transcript = unicodedata.normalize("NFKD", self._transcript.strip()) \
            .encode("ascii", "ignore") \
            .decode("ascii", "ignore")

    @property
    def filename(self):
        return self._filename

    @property
    def channel(self):
        return self._channel

    @property
    def speaker_id(self):
        return self._speaker_id

    @property
    def start_time(self):
        return self._start_time

    @property
    def stop_time(self):
        return self._stop_time

    @property
    def labels(self):
        return self._labels

    @property
    def transcript(self):
        return self._transcript


def parse_stm_file(stm_file):
    r"""
    For TED processing: Parses an STM file at ``stm_file`` into a list of :class:`STMSegment`.
    """
    stm_segments = []
    with codecs.open(stm_file, encoding="utf-8") as stm_lines:
        for stm_line in stm_lines:
            stmSegment = STMSegment(stm_line)
            if not "ignore_time_segment_in_scoring" == stmSegment.transcript:
                stm_segments.append(stmSegment)
    return stm_segments


def load_data_ted_dev(dir_path):
    """Load TEDLIUM all development data to memory for evaluation purpose."""
    if not path.exists(dir_path):
        raise ValueError('Development directory does not exist.')
    if not path.exists(path.join(dir_path, 'mfcc_segments')):
        raise ValueError('MFCC directory has not been created.')
    if not path.exists(path.join(dir_path, 'labels.csv')):
        raise ValueError('Label file has not been generated.')

    labels, recs, starts, ends = load_ted_labels_indices(path.join(dir_path, 'labels.csv'))
    num_samples = len(labels)
    dev_data = np.zeros((num_samples, hp.win_len, hp.mfcc_dim))
    dev_labels = np.zeros((num_samples,), dtype=int)

    prev_rec = ''
    mfcc_dir = path.join(dir_path, 'mfcc_segments')
    n = 0
    for label, rec, start, end in zip(labels, recs, starts, ends):
        if rec != prev_rec:
            with h5py.File(path.join(mfcc_dir, rec + '.hdf5'), 'r') as f:
                rec_features = f['mfcc'][:]  # Gives a numpy array
            prev_rec = rec

        dev_data[n] = rec_features[start: end]
        dev_labels[n] = label
        n += 1

    return dev_data, dev_labels


def load_ted_labels_indices(file_path):
    """Load the indexing csv file for TEDLIUM. Format: [label, recoding name, start index, end index]."""
    label_start_end = np.loadtxt(file_path, dtype=int, delimiter=',', usecols=[0, 2, 3])
    recs = np.loadtxt(file_path, dtype='S50', delimiter=',', usecols=[1]).astype(str)
    # default str dtype trucates longer strings.

    # Return in the same order as in format.
    return label_start_end[:, 0], recs, label_start_end[:, 1], label_start_end[:, 2]


def extract_mfcc(sig):
    # Extract MFCC feature according to http://voicebiometry.org/.

    # Extraction parameters taken from VBS.
    SOURCERATE = 1250
    TARGETRATE = 100000
    LOFREQ = 120
    HIFREQ = 3800

    ZMEANSOURCE = True
    WINDOWSIZE = 250000.0
    USEHAMMING = True
    PREEMCOEF = 0.97
    NUMCHANS = 24
    CEPLIFTER = 22
    NUMCEPS = 19
    ADDDITHER = 1.0
    RAWENERGY = True
    ENORMALISE = True

    deltawindow = accwindow = 2

    cmvn_lc = 150
    cmvn_rc = 150

    fs = 1e7 / SOURCERATE

    fbank_mx = features.mel_fbank_mx(winlen_nfft=WINDOWSIZE / SOURCERATE,
                                     fs=fs,
                                     NUMCHANS=NUMCHANS,
                                     LOFREQ=LOFREQ,
                                     HIFREQ=HIFREQ)

    if ADDDITHER > 0.0:
        sig = features.add_dither(sig, ADDDITHER)

    fea = features.mfcc_htk(sig,
                            window=WINDOWSIZE / SOURCERATE,
                            noverlap=(WINDOWSIZE - TARGETRATE) / SOURCERATE,
                            fbank_mx=fbank_mx,
                            _0='first',
                            NUMCEPS=NUMCEPS,
                            RAWENERGY=RAWENERGY,
                            PREEMCOEF=PREEMCOEF,
                            CEPLIFTER=CEPLIFTER,
                            ZMEANSOURCE=ZMEANSOURCE,
                            ENORMALISE=ENORMALISE,
                            ESCALE=0.1,
                            SILFLOOR=50.0,
                            USEHAMMING=True)

    # Add derivative.
    fea = features.add_deriv(fea, (deltawindow, accwindow))

    # Reshaping to SFeaCat convention
    fea = fea.reshape(fea.shape[0], 3, -1).transpose((0, 2, 1)).reshape(fea.shape[0], -1)

    # Apply floating CMVN.
    fea = features.cmvn_floating(fea, cmvn_lc, cmvn_rc, unbiased=True)

    return fea


def sampling_ted_triplet(labels, starts, ends, recs, feature_dir, batch_size=16):
    """ Prepare one batch containing anchor, postive and negative for triplet network.

    Inputs:
    all as numpy array.
    feature_dir: directory containing MFCC features for all recordings.
    """

    # Sample positive and negative speakers.
    num_speakers = len(np.unique(labels))
    pos_speaker, neg_speaker = np.random.choice(num_speakers, size=2, replace=False)

    # For both, sample segments.
    pos_indices = np.where(labels == pos_speaker)[0]
    neg_indices = np.where(labels == neg_speaker)[0]
    if len(pos_indices) < batch_size * 2:  # In case the recoding is very short.
        pos_indices = np.random.choice(pos_indices, size=batch_size * 2, replace=True)
    else:
        pos_indices = np.random.choice(pos_indices, size=batch_size * 2, replace=False)
    if len(neg_indices) < batch_size:
        neg_indices = np.random.choice(neg_indices, size=batch_size, replace=True)
    else:
        neg_indices = np.random.choice(neg_indices, size=batch_size, replace=False)

    # Get {recoding name : [[start index, end index]]} dictionary since one speaker can have multiple recordings.
    pos_dict, neg_dict = {}, {}
    for pos_index in pos_indices:
        pos_rec = recs[pos_index]
        if pos_rec not in pos_dict:
            pos_dict[pos_rec] = []
        pos_dict[pos_rec].append([starts[pos_index], ends[pos_index]])
    for neg_index in neg_indices:
        neg_rec = recs[neg_index]
        if neg_rec not in neg_dict:
            neg_dict[neg_rec] = []
        neg_dict[neg_rec].append([starts[neg_index], ends[neg_index]])

    # Load data for the batch based on the dictionaries.
    pos_batch = np.zeros((batch_size * 2, hp.win_len, hp.mfcc_dim))
    neg_batch = np.zeros((batch_size, hp.win_len, hp.mfcc_dim))
    n = 0
    for rec, indices in pos_dict.items():
        with h5py.File(path.join(feature_dir, rec + '.hdf5'), 'r') as f:
            rec_features = f['mfcc'][:]  # Gives a numpy array
        for start, end in indices:
            pos_batch[n] = rec_features[start: end]
            n += 1
    n = 0
    for rec, indices in neg_dict.items():
        with h5py.File(path.join(feature_dir, rec + '.hdf5'), 'r') as f:
            rec_features = f['mfcc'][:]
        for start, end in indices:
            neg_batch[n] = rec_features[start: end]
            n += 1

    np.random.shuffle(pos_batch)  # Shuffles the first axis.
    np.random.shuffle(neg_batch)
    return pos_batch[0: batch_size], pos_batch[batch_size:], neg_batch


def sampling_ted_batch(labels, starts, ends, recs, feature_dir, batch_size=64, speakers_per_batch=8):
    """Prepare one batch containing multiple speakers"""

    # Sample speakers.
    speakers = np.unique(labels)
    speaker_ids = np.random.choice(speakers, size=speakers_per_batch, replace=False)

    # For each speaker, sample segments.
    num_seg_per_speaker = batch_size // speakers_per_batch
    rec_dict = {}  # Indexing dictionary for recordings.
    for speaker_id in speaker_ids:
        indices = np.where(labels == speaker_id)[0]
        if len(indices) < num_seg_per_speaker:
            indices = np.random.choice(indices, size=num_seg_per_speaker, replace=True)  
        else:
            indices = np.random.choice(indices, size=num_seg_per_speaker, replace=False)

        # Get {recoding name : [[start index, end index, speaker ID]]} dictionary since one speaker can have
        # multiple recordings.
        for index in indices:
            rec = recs[index]
            if rec not in rec_dict:
                rec_dict[rec] = []
            rec_dict[rec].append([starts[index], ends[index], speaker_id])

    # Load data for the batch based on the dictionaries.
    batch = np.zeros((batch_size, hp.win_len, hp.mfcc_dim))
    batch_labels = np.zeros((batch_size,), dtype=int)
    n = 0
    for rec, values in rec_dict.items():
        with h5py.File(path.join(feature_dir, rec + '.hdf5'), 'r') as f:
            rec_features = f['mfcc']  # Gives a HDF5 object
            for start, end, label in values:
                # Naive slicing on hdf5 is much faster than converting it to np array first.
                batch[n] = rec_features[start: end]
                batch_labels[n] = label
                n += 1

    
    shuffle_ind = np.arange(batch_size)
    # np.random.shuffle(shuffle_ind)

    return batch[shuffle_ind], batch_labels[shuffle_ind]


def mean_score(test_pred, test_label, train_pred, train_label, prt=True):
    """ mean accuracy averaged on all classes.
    
    Inputs:
    -test_pred: predicted labels on testing data
    -test_label: ground-truth labels on testing data
    -train_pred: predicted labels on training data
    -train_label: ground-truth labels on training data
    
    Outputs:
    -C_tt_norm: normalized testing confusion matrix
    -C_norm: normalized training confusion matrix
    
    """
    C = confusion_matrix(train_label, train_pred)
    C_norm = 100 * np.transpose(np.transpose(C) / C.astype(np.float).sum(axis=1))

    C_tt = confusion_matrix(test_label, test_pred)
    C_tt_norm = 100 * np.transpose(np.transpose(C_tt) / C_tt.astype(np.float).sum(axis=1))

    if prt:
        print('Training fit rate : ' + str(np.mean(np.diag(C_norm))))
        print('Testing accuracy : ' + str(np.mean(np.diag(C_tt_norm))))

    return C_tt_norm, C_norm


def clustering_metrics(true, pred):
    """Evaluate clustering performance using NMI, clustering accuracy and purity."""

    # calculates normalized mutual information
    NMI = normalized_mutual_info_score(true, pred)

    # calculates clustering accuracy
    D = max(pred.max(), true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(true)):
        w[pred[i], true[i]] += 1
    ind = linear_assignment(w.max() - w)
    ACC = sum([w[i, j] for i, j in ind]) * 1.0 / len(true)

    # calculates purity score
    A = np.c_[(pred, true)]
    n_accurate = 0.
    for j in np.unique(A[:, 0]):
        z = A[A[:, 0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])
    purity = n_accurate / A.shape[0]

    print("NMI score: " + str(NMI))
    print("Clustering accuracy: " + str(ACC))
    print("Purity score: " + str(purity))

    return NMI, ACC, purity


def vis_tsne(rep, targets):
    """Visualize TSNE embedding with optional colorbar."""

    n_classes = len(np.unique(targets))

    f = plt.figure(figsize=(5, 5), dpi=100)
    plt.scatter(rep[:, 0], rep[:, 1], c=targets, cmap=plt.cm.get_cmap('tab20', n_classes), s=8, edgecolor='black',
                linewidth=0.05)  # Vega20 contains 20 colors.
    f.savefig('tsne.pdf', bbox_inches='tight')

    plt.show()
    
def adaptive_margin(anchor, positive, negative, margin1=0.8, extra=False):
    """Adaptive margin function for metric learning: For the triplet/quadruplet network.
    Args:
        anchor: anchor feature vectors of shape [batch_size, dim].
        positive: features of the same class as anchor.
        negative: features of the difference classes as anchor.
        margin1: horizon for negative examples.


    Output:
        margin: batch_wise margin as scalar
    """
    mean_pos = tf.reduce_mean(tf.reduce_sum(tf.square(anchor - positive), 1))

    mean_neg = tf.reduce_mean(tf.reduce_sum(tf.square(anchor - negative), 1))

    margin = mean_neg - mean_pos

    return tf.maximum(margin, margin1)
