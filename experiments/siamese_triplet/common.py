import numpy as np
from scipy.spatial.distance import cdist


def sample_episode(labels, voca_size, shots):
    vocabulary = np.unique(labels)

    ep_vocabulary = np.random.choice(vocabulary, size=voca_size, replace=False)

    ep_train_subset = []
    ep_test_subset = []
    for l in ep_vocabulary:
        where_label = np.random.permutation(np.where(labels == l)[0])
        ep_train_subset.extend(where_label[:shots])
        ep_test_subset.extend(where_label[shots:])

    return np.array(ep_train_subset), np.array(ep_test_subset)


def evaluate_knn(x_train, x_test, labels_train, labels_test, k):
    ep_vocabulary = np.unique(labels_train)

    dists = cdist(x_test, x_train, metric='cosine')

    neighbours = np.argsort(dists, axis=1)[:, :k]

    neighbours_labels = labels_train[None, neighbours][0]
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
