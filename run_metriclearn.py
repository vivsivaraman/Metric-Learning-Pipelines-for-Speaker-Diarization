""" Training the metric learning pipeline: """

import csv
import tensorflow as tf
import numpy as np
from os import path
from sklearn.cluster import KMeans
import sys

from hyperparams import Hyperparams as hp
import Models.attention_inference
import utils
import matplotlib.pyplot as plt


def eval_clustering(embs, label):
    # Evaluate clustering performance for the given embedding again the ground-truth label.

    kmeans = KMeans(n_clusters=len(np.unique(label)), init='k-means++', n_jobs=-1, random_state=0)
    kmeans.fit_predict(embs)

    # Get clustering evaluation metrics.
    metric_values = utils.clustering_metrics(label, kmeans.labels_)

    return metric_values


def main(args):
    # Overwrite parameters if given on command line.
    #if len(args) > 1:
    #    hp.triplet_margin = float(args[1])
    #    hp.num_speakers_per_batch = int(args[2])
    #    hp.logdir = 'experiments_%02d' % int(args[3])
    
    plot = True #Boolean whether to plot the overall loss versus the iterations 
    

    whole_set = True  # Whether training on the whole TEDLIUM set.

    with tf.Session() as sess:

        # Setup network.
        if(hp.metric_model == 'trip' or hp.metric_model == 'quad'):
            model = Models.attention_inference.Metriclearningmodel(loss_type=hp.loss_type, margin=hp.margin, sampling_type=hp.sampling_type,phase='train')
        else:
            raise ValueError('Model not defined.')
            
        print(hp.sampling_type,'  ',hp.loss_type, '  ',hp.margin)

        train_step = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8) \
            .minimize(model.loss, global_step=model.global_step)
        saver = tf.train.Saver(max_to_keep=None)  # Save all models.

        # Load labels and indices for sampling batches.
        train_dir = hp.tedlium_trainsubset_dir if not whole_set else hp.tedlium_trainsubset_dir
        labels, recs, start, end = utils.load_ted_labels_indices(path.join(train_dir, 'labels.csv'))
        # Load development data.
        if not whole_set:  # No development set in the whole set - parameters have already been tuned.
            dev_dir = path.join(path.dirname(train_dir), 'dev')
            dev_data, dev_label = utils.load_data_ted_dev(dev_dir)

        # Initialization.
        init = tf.global_variables_initializer()
        sess.run(init)
        summary_writer = tf.summary.FileWriter(hp.logdir)

        # Save all parameters.
        param_values = vars(hp)
        with open(path.join(hp.logdir, 'params.csv'), 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            for k, v in param_values.items():
                writer.writerow([k, v])

        # Start training.
        num_batches_per_epoch = len(labels) // hp.batch_size  # One equivalent epoch is finishing the sampling
        # of total number of segments in train set.
        print('Starting Training')
        loss_val = []
        overall_loss = []
        for epoch in range(1, hp.num_epochs + 1):
            # Process one epoch.

            print('EPOCH: ' + str(epoch))

            for step in range(num_batches_per_epoch):

                batch_data, batch_labels = utils.sampling_ted_batch(labels, start, end, recs,
                                                                    path.join(train_dir, 'mfcc_segments'),
                                                                    batch_size=hp.batch_size,
                                                                    speakers_per_batch=hp.num_speakers_per_batch)

                _, train_summ, global_step, l = sess.run([train_step, model.train_summary, model.global_step, model.loss], feed_dict={model.input: batch_data, model.y_: batch_labels})
                loss_val.append(l)  # Loss from every global step

                
                if global_step % 10 == 0:
                    summary_writer.add_summary(train_summ, global_step)

                if global_step % 200 == 0:
                    print('Global step: ' + str(global_step))
                    print('Global step loss',l)
                    if not whole_set:
                        # Evaluate on development set using clustering metrics.
                        print('Development Data Eval:')
                        dev_embs, dev_loss = model.get_embeddings(sess, dev_data, dev_label)
                        dev_metrics = eval_clustering(dev_embs, dev_label)

                        # Save development result on the run.
                        dev_info = [int(global_step)] + [dev_loss] + list(dev_metrics)
                        dev_file = path.join(hp.logdir, 'dev_history.csv')
                        append_or_write = 'a' if path.exists(dev_file) else 'w'
                        with open(dev_file, append_or_write) as f:
                            writer = csv.writer(f, lineterminator='\n')
                            writer.writerow(dev_info)
                        
                    # Save model every 200 global steps.
                    saver.save(sess, path.join(hp.logdir, 'model_step_%06d' % global_step))

            print('Mean Loss in EPOCH',np.mean(loss_val))  # Printing the average loss after every epoch
            overall_loss.append(loss_val)
            loss_val = []
        
        overall_loss = np.asarray(overall_loss)
        np.savetxt('train_loss.txt', overall_loss, delimiter=',')
        print('Training Completed')
        
        if plot:
            plt.plot(overall_loss)
            plt.xlabel('No. of global steps')
            plt.ylabel('Loss')
            plt.show()
            


if __name__ == '__main__':
    main(sys.argv)
