""" Network inference for architecture: MFCC + triplet with attention networks. """

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.contrib.losses import metric_learning
from os import path, makedirs

from hyperparams import Hyperparams as hp
from Models.loss_functions import loss_triplet, loss_quadruplet, semihard_quadruplet
from Models.sampling_methods import distance_weighted_sampling, random_sampling, batch_all
from Models.modules_attention import input_embedding, embedding, multihead_attention, feedforward
import utils

class EmbeddingModel:
    """The embedding network for all inputs."""

    @staticmethod
    def embed(x, phase):
        """Embed all give tensors. Use similar network structure as in the time series project."""

        is_train = True if phase == 'train' else False

        # Input embedding: convert input vector to dimension of hp.hidden_units.
        embs = input_embedding(x, num_units=hp.hidden_units, embed_type=hp.embed_type)
        print('Size after input embedding: ', embs.get_shape())

        # Positional Encoding.
        embs += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(x)[1]), 0), [tf.shape(x)[0], 1]),
                          vocab_size=hp.win_len, num_units=hp.hidden_units,
                          zero_pad=False, scale=False, scope="enc_pe")
        print("Size after positional encoding: ", embs.get_shape())

        # Attention blocks.
        for i in range(hp.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                # Multi-head Attention
                embs = multihead_attention(queries=embs, keys=embs, num_units=hp.hidden_units,
                                           num_heads=hp.num_heads, dropout_rate=hp.dropout_rate,
                                           is_training=is_train, causality=False)

                # Feed Forward
                embs = feedforward(embs, num_units=[2 * hp.hidden_units, hp.hidden_units])
        print("Size after multi-head_attention: ", embs.get_shape())

        # Temporal pooling by averaging on the time dimension.
        embs = tf.reduce_mean(embs, axis=1)

        return embs


class Metriclearningmodel(EmbeddingModel):
    """Triplet / Quadruplet network for metric learning."""

    # Create model.
    def __init__(self, loss_type, margin, sampling_type, phase='train'):
        
        self.input = tf.placeholder(tf.float32, shape=(hp.batch_size, hp.win_len, hp.mfcc_dim))
        self.y_ = tf.placeholder(tf.int64, [None])
        
        self.loss_type = loss_type
        self.margin = margin
        self.sampling_type = sampling_type
        self.phase = phase

        # Embed them.
        self.emb = self.embed(self.input, self.phase)
        
        if self.sampling_type == 'dws':
            self.emb_a, self.emb_p, self.emb_n, self.emb_n1 = distance_weighted_sampling(self.emb, hp.batch_k, hp.cutoff, hp.nonzero_loss_cutoff)

        elif self.sampling_type == 'random':
            self.emb_a, self.emb_p, self.emb_n, self.emb_n1 = random_sampling(self.emb, hp.batch_k, hp.cutoff)

        elif self.sampling_type == 'semihard': 
            self.emb_a, self.emb_p, self.emb_n, self.emb_n1 = batch_all(self.emb)
        else:
            raise ValueError('Sampling type not defined.')

        #Margin selection
        if self.margin == 'fixed':
            self.trip_margin = hp.triplet_margin
            self.quad_margin = hp.quadruplet_margin

        elif self.margin == 'adaptive':
            self.trip_margin = utils.adaptive_margin(self.emb_a, self.emb_p, self.emb_n, hp.triplet_margin)
            self.quad_margin = 0.5*self.trip_margin
        
        else:
            raise ValueError('Margin type not defined.')

        # Compute the loss.
        if self.loss_type == 'triplet':
            if self.sampling_type == 'semihard':
                self.emb_normalized = tf.nn.l2_normalize(self.emb, axis=-1)  # Note the axis to normalize.
                self.loss = metric_learning.triplet_semihard_loss(self.y_, self.emb_normalized, margin=self.trip_margin)
            else:                
                self.loss = loss_triplet(self.emb_a, self.emb_p, self.emb_n, self.trip_margin)
        
        elif self.loss_type == 'quadruplet':
            if self.sampling_type == 'semihard':                
                self.emb_normalized = tf.nn.l2_normalize(self.emb, axis=-1)  # Note the axis to normalize.
                self.loss = semihard_quadruplet(self.emb_a, self.emb_p, self.emb_n, self.emb_n1, self.quad_margin) + \
                            metric_learning.triplet_semihard_loss(self.y_, self.emb_normalized, margin=self.trip_margin)
    
            else:
                self.loss = loss_quadruplet(self.emb_a, self.emb_p, self.emb_n, self.emb_n1, self.quad_margin) + \
                            loss_triplet(self.emb_a, self.emb_p, self.emb_n, self.trip_margin)
        else:
            raise ValueError('Type of loss not defined.')

        # Get model summary and save model parameters.
        self.train_summary = tf.summary.scalar('train_loss', self.loss)
        self.saver = tf.train.Saver(var_list=None, max_to_keep=0)  # Save all variables.
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

    # Compute embeddings from the model under a session, for another set of data and label.
    def get_embeddings(self, sess, data, label, save=False, save_path=''):

        # If number of batches is not in multiples of 3, pad zero batches.
        n_padded = hp.batch_size - len(data) % hp.batch_size
        if n_padded != 0:
            data = np.concatenate((data, np.zeros((n_padded, data.shape[1], data.shape[2]))), axis=0)
            label = np.concatenate((label, np.zeros((n_padded,), dtype=int)), axis=0)

        num_batches = len(data) // hp.batch_size
        embs = []
        mean_loss = 0  # Averaged loss across all batches.
        for i in range(num_batches):
            batch_data = data[i * hp.batch_size: (i + 1) * hp.batch_size]
            batch_label = label[i * hp.batch_size: (i + 1) * hp.batch_size]

            # Generate embeddings and perform clustering.
            batch_embs, batch_loss = sess.run([self.emb, self.loss],
                                              feed_dict={self.input: batch_data, self.y_: batch_label})
            embs.append(batch_embs)
            mean_loss += batch_loss / float(num_batches)

        embs = np.concatenate(embs, axis=0)
        # Throw away embeddings for padded batches.
        if n_padded != 0:
            embs = embs[0:len(embs) - n_padded]

        if save:
            assert save_path != ''
            assert path.exists(path.dirname(save_path))
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('embs', data=embs)

        return embs, mean_loss



