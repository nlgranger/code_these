import numpy as np
import seqtools
from theano import tensor as T
import scipy.spatial.distance

from sltools.nn_utils import adjust_length, cdist, logsumexp, log_softmax, \
    categorical_crossentropy_logdomain


def sample_triplets(labels, recordings, vocabulary, n):
    where_labels = {l: np.where(labels == l)[0] for l in vocabulary}
    where_not_labels = {l: np.where(labels != l)[0] for l in vocabulary}

    triplets = np.empty((n, 3), dtype=np.int32)

    triplets[:, 0] = labels[np.arange(n) % len(labels)]

    for i in range(n):
        wl = where_labels[triplets[i, 0]]
        compatible = recordings[wl] != recordings[triplets[i, 0]]
        wl = wl[compatible]
        triplets[i, 1] = np.random.choice(wl)

    for i in range(n):
        triplets[i, 2] = np.random.choice(where_not_labels[triplets[i, 0]])

    return np.random.permutation(triplets)


def sample_episode(labels, recordings, vocabulary, voca_size, shots, test_size=None):
    """Sample a training episode.

    :param labels:
        labels from the training set
    :param recordings:
        indexes shared by all variations of an augmented sequence
    :param vocabulary:
        unique labels
    :param voca_size:
        number of classes in one episode (must be smaller than vocabulary)
    :param shots:
        number of training samples for each classes in one episode.
    :param test_size:
        number of test samples in the episode
    :return:
        A tuple containing:

        - indexes of voca_size by shots training samples
        - test_size indexes of testing samples
        - their labels (within the episode, therefore in [0, voca_size - 1])
    """
    ep_vocabulary = np.random.choice(vocabulary, size=voca_size, replace=False)

    ep_train_subset = []
    ep_test_subset = []
    for l in ep_vocabulary:
        where_label = np.random.permutation(np.where(labels == l)[0])
        non_redundants = np.invert(np.isin(recordings[where_label[shots:]],
                                           recordings[where_label[:shots]]))
        ep_train_subset.extend(where_label[:shots])
        ep_test_subset.extend(where_label[shots:][non_redundants])

    ep_train_subset = np.array(ep_train_subset)
    if test_size is not None:
        ep_test_subset = np.random.choice(ep_test_subset, size=test_size, replace=False)

    ep_test_labels = np.argmax(labels[ep_test_subset, None] == ep_vocabulary, axis=1)
    ep_test_labels = ep_test_labels.astype(np.int32)

    return ep_train_subset, ep_test_subset, ep_test_labels


def triplets2minibatches(feat_seqs, durations, triplets, batch_size, max_time):
    feat_seqs = [seqtools.smap(lambda s: adjust_length(s, max_time), f)
                 for f in feat_seqs]
    durations = np.fmin(durations, max_time)
    triplets = np.asarray(triplets)

    # maximum number of triplet elements by batch
    actual_batch_size = batch_size - batch_size % 3
    n_batches = len(triplets) * 3 // actual_batch_size

    flat_triplets = np.ravel(triplets)[:n_batches * actual_batch_size]  # drop last batch

    batched_indexes = np.zeros((n_batches, batch_size), dtype=triplets.dtype)
    batched_indexes[:, :actual_batch_size] = np.reshape(
        flat_triplets, (n_batches, actual_batch_size))

    feat_batches = [
        seqtools.smap(
            lambda b, f=f: np.stack(seqtools.gather(f, b), axis=0),
            batched_indexes)
        for f in feat_seqs]
    duration_batches = durations[batched_indexes]

    return seqtools.collate(feat_batches + [duration_batches])


def episode2minibatch(episode, feat_seqs, durations, max_time):
    ep_train_subset, ep_test_subset, ep_labels = episode
    episode_feats = [
        np.stack([adjust_length(f[i], max_time)
                  for i in np.concatenate([ep_train_subset, ep_test_subset])])
        for f in feat_seqs]

    ep_durations = np.fmin(durations[np.concatenate([ep_train_subset, ep_test_subset])],
                           max_time)

    return tuple([*episode_feats, ep_durations, ep_labels])


def triplet_loss(linout, delta=1., metric='euclidean', clip=1.0, p=None):
    left = linout[0::3]
    middle = linout[1::3]
    right = linout[2::3]

    if metric == 'squared':
        dist = T.pow((left - middle).norm(2, axis=1), 2.0) \
            - T.pow((left - right).norm(2, axis=1), 2.0)
    elif metric == 'euclidean':
        dist = (left - middle).norm(2, axis=1) \
               - (left - right).norm(2, axis=1)
    elif metric == 'minkowski':
        dist = (left - middle).norm(p, axis=1) \
               - (left - right).norm(p, axis=1)
    elif metric == 'cosine':
        norm_left = left.norm(2, axis=1)
        norm_middle = middle.norm(2, axis=1)
        norm_right = right.norm(2, axis=1)
        dist = T.sum(
            left * (right * norm_middle[:, None] - middle * norm_right[:, None]),
            axis=1) / (norm_left * norm_middle * norm_right + 0.0001)
    else:
        raise NotImplementedError()

    return T.minimum(T.maximum(dist + delta, 0), clip)


def kernel_loss(linout, targets_var, ep_voca_size, shots):
    ep_train_embeddings = linout[:ep_voca_size * shots]
    ep_test_embeddings = linout[ep_voca_size * shots:]

    alpha = 1 - cdist(ep_test_embeddings, ep_train_embeddings, metric='cosine')
    alpha = log_softmax(alpha, axis=1)
    # alpha -= logsumexp(alpha, axis=1, keepdims=True)  # instead of softmax
    alpha = logsumexp(
        T.reshape(alpha, (ep_test_embeddings.shape[0], ep_voca_size, shots)),
        axis=2)

    train_loss = 0
    target_loss = categorical_crossentropy_logdomain(alpha, targets_var)
    train_loss += target_loss.mean()
    regularizer = T.pow(1.0 - linout.norm(2, axis=1), 2)
    train_loss += 0.1 * regularizer.mean()

    return train_loss


def evaluate_knn(x_train, x_test, labels_train, labels_test, k):
    ep_vocabulary = np.unique(labels_train)

    dists = scipy.spatial.distance.cdist(x_test, x_train, metric='cosine')

    neighbours = np.argsort(dists, axis=1)[:, :k]

    neighbours_labels = labels_train[neighbours]
    neighbours_dists = dists[np.arange(len(labels_test))[:, None], neighbours]

    stats = np.empty((len(labels_test), len(ep_vocabulary)),
                     dtype=[('freq', 'i4'), ('dist_score', 'f4'), ('class', 'i4')])
    for i, l in enumerate(ep_vocabulary):
        stats['freq'][:, i] = np.sum(neighbours_labels == l, axis=1)
        stats['dist_score'][:, i] = -np.sum(neighbours_dists * (neighbours_labels == l),
                                            axis=1)
        stats['class'][:, i] = l

    stats = np.sort(stats, axis=1)

    ranks = len(ep_vocabulary) - 1 \
        - np.argmax(labels_test[:, None] == stats['class'], axis=1)

    return ranks


def evaluate_matching(x_train, x_test, labels_train, labels_test, ep_voca_size, shots):
    ep_vocabulary = labels_train[::shots]
    assert len(ep_vocabulary) == ep_voca_size, \
        "{} != {}".format(len(ep_vocabulary), ep_voca_size)

    alpha = 1 - scipy.spatial.distance.cdist(x_test, x_train, metric='cosine')
    alpha = np.exp(alpha) / np.sum(np.exp(alpha), axis=1)[:, None]
    # alpha = alpha / np.sum(alpha, axis=1)[:, None]
    alpha = np.sum(
        np.reshape(alpha, (x_test.shape[0], ep_voca_size, shots)),
        axis=2)

    ep_test_labels = np.argmax(labels_test[:, None] == ep_vocabulary[None, :], axis=1)
    ranks = np.argsort(alpha, axis=1)
    ranks = np.argmax(ranks == ep_test_labels[:, None], axis=1)
    return ep_voca_size - 1 - ranks
