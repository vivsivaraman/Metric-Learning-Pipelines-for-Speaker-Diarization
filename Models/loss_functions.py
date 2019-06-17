
import tensorflow as tf

def loss_triplet(anchor, positive, negative, margin=0.8, extra=False, scope='triplet_loss'):
    """Loss function for metric learning: triplet loss for triplet network.
    Args:
        anchor: anchor feature vectors of shape [batch_size, dim].
        positive: features of the same class as anchor.
        negative: features of the difference classes as anchor.
        margin: horizon for negative examples.
        extra: also return distances for positive and negative.
        scope: tensor scope.

    Output:
        loss: triplet loss as scalar and optionally average_pos_dist, average_neg_dist.
    """

    eps = 1e-10
    with tf.name_scope(scope):
        d_pos = tf.reduce_sum(tf.square(anchor - positive), 1)
        d_neg = tf.reduce_sum(tf.square(anchor - negative), 1)

        # loss = tf.reduce_sum(tf.maximum(0., margin + d_pos - d_neg))
        loss = tf.reduce_mean(tf.maximum(0., margin + d_pos - d_neg))

        if extra:
            pos_dist = tf.reduce_mean(tf.sqrt(d_pos + eps), name='pos-dist')
            neg_dist = tf.reduce_mean(tf.sqrt(d_neg + eps), name='neg-dist')
            return loss, pos_dist, neg_dist
        else:
            return loss


def loss_quadruplet(anchor, positive, negative1, negative2, margin=0.8, extra=False, scope='quadruplet_loss'):
    """Loss function for metric learning: quadruplet loss for quadruplet network.
    Args:
        anchor: anchor feature vectors of shape [batch_size, dim].
        positive: features of the same class as anchor.
        negative1: features of the difference classes as anchor.
        negative2: features of the difference classes as anchor as well as negative1.
        margin: horizon for negative examples.
        extra: also return distances for positive and negative.
        scope: tensor scope.

    Output:
        loss: quadruplet loss as scalar and optionally average_pos_dist, average_neg_dist.
    """

    eps = 1e-10
    with tf.name_scope(scope):
        d_pos = tf.reduce_sum(tf.square(anchor - positive), 1)
        d_neg = tf.reduce_sum(tf.square(negative1 - negative2), 1)

        # loss = tf.reduce_sum(tf.maximum(0., margin + d_pos - d_neg))
        loss = tf.reduce_mean(tf.maximum(0., margin + d_pos - d_neg))

        if extra:
            pos_dist = tf.reduce_mean(tf.sqrt(d_pos + eps), name='pos-dist')
            neg_dist = tf.reduce_mean(tf.sqrt(d_neg + eps), name='neg-dist')
            return loss, pos_dist, neg_dist
        else:
            return loss

def semihard_quadruplet(a,p,n,n1,margin):

    d_pos = tf.reduce_sum(tf.square(a - p), 1)
    d_neg = tf.reduce_sum(tf.square(n - n1), 1)

    loss = d_pos - d_neg + margin 
    loss = tf.reshape(loss, [-1])

    semihard_negatives = tf.where(tf.logical_and(loss < margin, loss > 0))

    loss = tf.gather(loss, semihard_negatives)
    loss = tf.reshape(loss, [-1])
    loss_val = tf.reduce_mean(loss)

    return loss_val

