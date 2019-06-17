import numpy as np
import tensorflow as tf

from hyperparams import Hyperparams as hp


def _pairwise_distances(embeddings):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    return distances


def distance_weighted_sampling(embeddings, batch_k, cutoff=0.5, nonzero_loss_cutoff=1.4):
    """Distance weighted sampling. See "sampling matters in deep embedding learning"
    paper for details.
    Refer "R Manmatha, Chao-Yuan Wu, Alexander J Smola, and
    Philipp Krähenbühl, “Sampling matters in deep embed-
    ding learning,” in Computer Vision (ICCV), 2017 IEEE
    International Conference on. IEEE, 2017, pp. 2859–
    2867"
    
    USING A MODIFIED VERSION from:
    "https://github.com/apache/incubator-mxnet/blob/master/example/gluon/embedding_learning/model.py"
    Parameters 
    ----------
    batch_k : int
        Number of segments per speaker class.
    Inputs:
        - **embeddings**: input tensor with shape (batch_size, embed_dim).
        Here we assume the consecutive batch_k examples are of the same class.
        For example, if batch_k = 5, the first 5 examples belong to the same class,
        6th-10th examples belong to another class, etc.
    Outputs:
        - x[a_indices]: sampled anchor embeddings.
        - x[p_indices]: sampled positive embeddings.
        - x[n_indices]: sampled negative embeddings.
        - x[n1_indices]: sampled negative1 embeddings.

    """
    # We sample only from negatives that induce a non-zero loss.
    # These are negatives with a distance < nonzero_loss_cutoff.
    # With a margin-based loss, nonzero_loss_cutoff == margin + beta. This is for the margin based loss given in the
    # sampling matters paper.

    embeddings = tf.nn.l2_normalize(embeddings, axis=-1)  # Note the axis to normalize.
    d = tf.to_float(tf.shape(embeddings)[1])

    distance = _pairwise_distances(embeddings)

    # Cut off to avoid high variance.
    distance = tf.maximum(distance, cutoff)

    # Subtract max(log(distance)) for stability.
    log_weights = ((2.0 - d) * tf.log(distance) - ((d - 3.0) / 2.0) * tf.log(1.0 - 0.25 * tf.square(distance)))
    log_weights = tf.cast(log_weights, tf.float64)

    weights = tf.exp((log_weights - tf.reduce_max(log_weights)))
    print(weights)

    # Sample only negative examples by setting weights of the same-class examples to 0.
    mask = np.ones((hp.batch_size, hp.batch_size))
    for i in range(0, hp.batch_size, batch_k):
        mask[i:i + batch_k, i:i + batch_k] = 0  # This ensures that the anchors and positives are not considered
        # during distance weighting. Distance weighting is only for negative examples

    mask_uniform_probs = mask * (1.0 / (hp.batch_size - batch_k))  # Uniform probability distribution
    mask_uniform_probs = tf.convert_to_tensor(mask_uniform_probs)

    mask = tf.to_float(tf.convert_to_tensor(mask))
    mask = tf.cast(mask, tf.float64)
    temp = tf.cast(tf.to_float(distance < nonzero_loss_cutoff), tf.float64)  # Taking only those distances < 1.4.
    # It is an assumption made by the sampling matters paper that if the distances fall below this value, they
    # will induce a non zero loss

    # The distance weighting process. The weights are computed with respect to the anchors.
    weights = weights * mask * temp
    weights_sum = tf.reduce_sum(weights, axis=1, keepdims=True)
    weights = weights / weights_sum

    np_weights = weights
    print(np_weights)

    a_indices, p_indices, n_indices, n1_indices = [], [], [], []
    elems = tf.range(0, hp.batch_size, 1)

    for i in range(hp.batch_size):
        block_idx = i // batch_k

        if weights_sum[i] != 0:

            samples = tf.multinomial(tf.log([np_weights[i]]), num_samples=batch_k - 1)  # note log-prob.
            # This is analogous to np.random.choice. Instead of taking the probability mass function,
            # this takes in the log probabilities.
            # It provides batch_k -1 samples based upon the probabilities given.
            # However, the usual tf.multinomial function takes in unnormalized log probabilities. But here I have given
            # normalized probabilities. I have checked this separately along with np.random.choice and both give me
            # similar results.
            # We here sample batch_k -1 samples because our hp.batch_size = 256, no. of speakers per segment is 64.
            # Therefore our batch_k = 4. To get the positives different from the current anchor,
            # we draw batch_k-1 samples.

            temp1 = tf.reshape(tf.cast(samples, tf.int32), [-1])
            out = tf.gather(elems, temp1)
            n_indices = tf.concat([n_indices, out], 0)  # Concatenating the negative indices.

        else:

            samples = tf.multinomial(tf.log([mask_uniform_probs[i]]), num_samples=batch_k - 1)  # note log-prob
            temp1 = tf.reshape(tf.cast(samples, tf.int32), [-1])
            out = tf.gather(elems, temp1)
            n_indices = tf.concat([n_indices, out], 0)

        for j in range(block_idx * batch_k, (block_idx + 1) * batch_k):

            j_tensor = tf.constant([j])

            if j != i:
                i_tensor = tf.constant([i])
                a_indices = tf.concat([a_indices, i_tensor], 0)  # Getting the anchor and positive segments.
                p_indices = tf.concat([p_indices, j_tensor], 0)  # It is assumed that batch_k examples are examples
                # from the same class.

    a_indices = tf.reshape(tf.cast(a_indices, tf.int32), [-1])
    p_indices = tf.reshape(tf.cast(p_indices, tf.int32), [-1])
    n_indices = tf.reshape(tf.cast(n_indices, tf.int32), [-1])

    # The following steps are used to determine the second set of negative classes for a given anchor, positive and
    # negative class
    # Here I have considered every element in n_indices to be an anchor and found the negative class for that value.
    # I have also ensured that the original anchors and positives are also not considered while sampling for the second
    # negative class
    z = 0
    cnt = 0
    for val in range((batch_k - 1) * hp.batch_size):  # Traversing 3* 256
        block_id = z // batch_k

        np_weights1 = tf.identity(np_weights)  # making a copy of np_weights
        # np_weights1 = np_weights
        # np_weights1 = tf.placeholder(tf.float64, [16, 256])
        var1 = n_indices[val]  # Making first element of the determined negatives to be the "anchor"

        var2 = tf.logical_not(tf.greater_equal(elems, block_id * batch_k))
        var3 = tf.logical_not(tf.less(elems, (block_id + 1) * batch_k))
        var3 = tf.cast(tf.logical_or(var2, var3),
                       tf.float64)  # These three steps logically ensure that when we sample a new negative,
        # the original anchor and postive class are not involved.
        # Also the current negative class is not involved
        np_weights1 = tf.multiply(np_weights1, var3)
        weights_sum1 = tf.reduce_sum(np_weights1, axis=1, keepdims=True)
        np_weights1 = (np_weights1 / (weights_sum1 + 1e-16))  # Normalizing the weights

        mask_uniform_probs1 = tf.identity(mask_uniform_probs)
        # mask_uniform_probs1 = mask_uniform_probs
        mask_uniform_probs1 = tf.multiply(mask_uniform_probs1, var3)

        weights_sum2 = tf.reduce_sum(mask_uniform_probs1, axis=1, keepdims=True)

        mask_uniform_probs1 = (mask_uniform_probs1 / (weights_sum2 + 1e-16))

        if weights_sum1[var1] != 0:

            samples = tf.multinomial(tf.log([np_weights1[var1]]), num_samples=1, seed=None)  # note log-prob
            temp1 = tf.reshape(tf.cast(samples, tf.int32), [-1])
            out = tf.gather(elems, temp1)
            n1_indices = tf.concat([n1_indices, out], 0)
            cnt += 1

        else:

            samples = tf.multinomial(tf.log([mask_uniform_probs1[var1]]), num_samples=1, seed=None)  # note log-prob
            temp1 = tf.reshape(tf.cast(samples, tf.int32), [-1])
            out = tf.gather(elems, temp1)
            n1_indices = tf.concat([n1_indices, out], 0)
            cnt += 1

        if cnt == 3:
            cnt = 0
            z += 1

    n1_indices = tf.reshape(tf.cast(n1_indices, tf.int32), [-1])

    embeddings_a_indices = tf.gather(embeddings, a_indices)
    embeddings_p_indices = tf.gather(embeddings, p_indices)
    embeddings_n_indices = tf.gather(embeddings, n_indices)
    embeddings_n1_indices = tf.gather(embeddings, n1_indices)
    return embeddings_a_indices, embeddings_p_indices, embeddings_n_indices, embeddings_n1_indices

def random_sampling(embeddings, batch_k, cutoff=0.5):
        """Random sampling. 
        """
        # We sample only from negatives that induce a non-zero loss.
        # These are negatives with a distance < nonzero_loss_cutoff.
        
        # Sample only negative examples by setting weights of the same-class examples to 0.
        mask = np.ones((hp.batch_size, hp.batch_size))

        for i in range(0, hp.batch_size, batch_k):
            mask[i:i + batch_k, i:i + batch_k] = 0

        mask_uniform_probs = mask * (1.0 / (hp.batch_size - batch_k))
        mask_uniform_probs = tf.convert_to_tensor(mask_uniform_probs)

        #Empty lists to store the indices
        t1 = [] #Negative 1
        t2 = [] #Anchor
        t3 = [] #Positive
        t4 = [] #Negative 2
        
        elems = tf.range(0, hp.batch_size, 1)

        for i in range(hp.batch_size):

            block_idx = i // batch_k

            #Drawing batch_k -1 samples uniformly 
            samples = tf.multinomial(tf.log([mask_uniform_probs[i]]), num_samples=batch_k-1, seed=None)  # note log-prob
            
            temp1 = tf.reshape(tf.cast(samples, tf.int32), [-1])
            out = tf.gather(elems, temp1)
            
            #Concatenating the 
            t1 = tf.concat([t1, out], 0)

            for j in range(block_idx * batch_k, (block_idx + 1) * batch_k):
                j_tensor = tf.constant([j])
                if j != i:
                    i_tensor = tf.constant([i])
                    t2 = tf.concat([t2, i_tensor], 0)
                    t3 = tf.concat([t3, j_tensor], 0)

        t1 = tf.reshape(tf.cast(t1, tf.int32), [-1])
        t2 = tf.reshape(tf.cast(t2, tf.int32), [-1])
        t3 = tf.reshape(tf.cast(t3, tf.int32), [-1])

        z = 0
        cnt = 0
        for val in range(hp.batch_size*(batch_k-1)):
            block_id = z // batch_k

            var1 = t1[val]
            var2 = tf.logical_not(tf.greater_equal(elems, block_id * batch_k))
            var3 = tf.logical_not(tf.less(elems, (block_id + 1) * batch_k))
            var3 = tf.cast(tf.logical_or(var2, var3), tf.float64)


            mask_uniform_probs1 = tf.identity(mask_uniform_probs)
            mask_uniform_probs1 = tf.multiply(mask_uniform_probs1, var3)

            weights_sum2 = tf.reduce_sum(mask_uniform_probs1, axis=1, keepdims=True)

            mask_uniform_probs1 = (mask_uniform_probs1 / (weights_sum2 + 1e-16))

            samples = tf.multinomial(tf.log([mask_uniform_probs1[var1]]), num_samples=1, seed=None)  # note log-prob
            # y = elems[tf.cast(samples[0][0], tf.int32)]
            temp1 = tf.reshape(tf.cast(samples, tf.int32), [-1])
            out = tf.gather(elems, temp1)
            # elems[tf.cast(samples[0][0], tf.int32)].eval()
            t4 = tf.concat([t4, out], 0)
            cnt += 1

            if cnt == 3:
                cnt = 0
                z += 1

        t4 = tf.reshape(tf.cast(t4, tf.int32), [-1])

        anchor = tf.gather(embeddings, t2)
        positive = tf.gather(embeddings, t3)
        n1 = tf.gather(embeddings, t1)
        n2 = tf.gather(embeddings, t4)
        return anchor, positive, n1, n2
    
    
def batch_all(embeddings):
    #Creating empty lists to store the indices of the anchor,positive and negatives
    anc = []
    pos = []
    neg = []
    neg1 = []
    
    #batch_k specifies the number of examples per class in order
    batch_k = hp.batch_k
    
    for i in range(hp.batch_size):
        block_idx = i // batch_k     #Accessing that particular block
        
        #The following code provides the the exhaustive list of possible, not redundant
        #quadruplets used for the loss calculation
        for j in range(block_idx * batch_k, (block_idx + 1) * batch_k):
            for k in range(hp.batch_size):
                for l in range(hp.batch_size):
                    if j != i and i-j !=1:    
                        if k not in range(block_idx * batch_k, (block_idx + 1) * batch_k):
                            if l not in range(block_idx * batch_k, (block_idx + 1) * batch_k) and l not in range((block_idx + int(k/batch_k)) * batch_k, (block_idx + int(k/batch_k)+ 1) * batch_k) :
                                anc.append(i)
                                pos.append(j)
                                neg.append(k)
                                neg1.append(l)
    
    #Converting the lists to tensors
    anc = tf.convert_to_tensor(np.asarray(anc))
    pos = tf.convert_to_tensor(np.asarray(pos))
    neg = tf.convert_to_tensor(np.asarray(neg))
    neg1 = tf.convert_to_tensor(np.asarray(neg1))
    
    #Extracting the embeddings for the corresponding indices
    anc = tf.gather(embeddings,anc)
    pos = tf.gather(embeddings,pos)
    neg = tf.gather(embeddings,neg)
    neg1 = tf.gather(embeddings,neg1)
    
    return anc,pos,neg,neg1





