import numpy as np
from .utils import gloss2seq, seq2gloss
from .nn_utils import jaccard, onehot


def filter_longshort(preds, boundaries, none_class=0):
    return gloss2seq([(g, start, stop) for (g, start, stop) in seq2gloss(preds)
                      if boundaries[0] <= (stop - start) < boundaries[1]],
                     len(preds), none_class)


def optimize_boundaries(targets, preds, vocabulary, search_boundaries):
    vocabulary = np.asarray(vocabulary)
    durations = np.array([len(t) for t in targets])
    short_max = search_boundaries[0] or max(durations)
    long_min = search_boundaries[1] or min(durations)
    long_max = search_boundaries[2] or max(durations)

    # short sequences
    thresholds = np.arange(0, short_max)
    jis = np.empty((len(thresholds),))
    for i, t in enumerate(thresholds):
        preds2 = [filter_longshort(p, (t, long_min), -1) for p in preds]
        jis[i] = np.mean([jaccard(onehot(l, vocabulary), onehot(p, vocabulary))
                          for l, p in zip(targets, preds2)])

    thres1 = thresholds[np.argmax(jis)]

    # long sequences
    thresholds = np.arange(long_min, long_max, 5)
    jis = np.empty((len(thresholds),))
    for i, t in enumerate(thresholds):
        preds2 = [filter_longshort(p, (0, t), -1) for p in preds]
        jis[i] = np.mean([jaccard(onehot(l, vocabulary), onehot(p, vocabulary))
                          for l, p in zip(targets, preds2)])
    thres2 = thresholds[np.argmax(jis)]

    return thres1, thres2
