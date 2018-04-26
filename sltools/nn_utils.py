import logging
import numpy as np
from theano import tensor as T
import lasagne
from lasagne.layers import Layer
from sklearn.metrics import confusion_matrix


# Generic mathematical functions --------------------------------------------------------

def logsumexp(x, axis=-1, keepdims=False):
    axis = x.ndim + axis if axis < 0 else axis
    k = T.max(x, axis=axis, keepdims=True)
    res = T.log(T.sum(T.exp(x - k), axis=axis, keepdims=True)) + k

    if keepdims:
        return res
    else:
        return res.dimshuffle([i for i in range(x.ndim) if i != axis])


def log_softmax(x, axis=-1):
    xdev = x - x.max(axis=axis, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=axis, keepdims=True))


def categorical_crossentropy_logdomain(log_predictions, targets):
    """Compute categorical cross entropy given predictions in log-domain along last axis.
    """
    data_shape = targets.shape
    print()
    log_predictions = T.reshape(log_predictions, (-1, log_predictions.shape[-1]))
    targets = T.flatten(targets)
    return T.reshape(-log_predictions[T.arange(targets.shape[0]), targets],
                     data_shape)


def onehot(y, labels):
    idx = np.empty((labels.max() + 1,), dtype=np.int32)
    for i, l in enumerate(labels):
        idx[l] = i
    labels = set(labels)
    res = np.full((len(y), len(labels)), False, dtype=np.bool)
    for i, l in enumerate(y):
        if l in labels:
            res[i, idx[l]] = True

    return res


def jaccard(y_true, y_pred):
    intersection = (y_pred * y_true).sum(axis=0)
    union = (y_pred | y_true).sum(axis=0)

    if not any(union):
        logging.info("empty sequence detected as empty")
        return 1
    else:
        return np.mean(intersection[union > 0] / union[union > 0])


def seq_hinge_loss(linout, targets, masks, weights=None, delta=1.):
    batch_size, max_len, voca_size = linout.shape

    if weights is None:
        weights = T.ones((voca_size,))
    else:
        weights = T.as_tensor_variable(weights).astype('floatX')

    logits = T.exp(log_softmax(linout))

    return T.reshape(
        lasagne.objectives.multiclass_hinge_loss(
            T.reshape(logits, (batch_size * max_len, voca_size)),
            T.flatten(targets), delta=delta) * weights[T.flatten(targets)],
        (batch_size, max_len)) * masks


def seq_ce_loss(linout, targets, masks, weights):
    batch_size, max_len, nlabels = linout.shape

    if weights is None:
        weights = T.ones((nlabels,))
    else:
        weights = T.as_tensor_variable(weights).astype('floatX')

    return T.reshape(
        categorical_crossentropy_logdomain(
            T.reshape(log_softmax(linout), (batch_size * max_len, nlabels)),
            T.flatten(targets)) * weights[T.flatten(targets)],
        (batch_size, max_len)) * masks


def cdist(a, b, metric='euclidean', p=2., epsilon=1e-4):
    if metric == 'cityblock':
        a = T.reshape(a, (a.shape[0], -1)).dimshuffle(0, 'x', 1)
        b = T.reshape(b, (b.shape[0], -1)).dimshuffle('x', 0, 1)
        return (a - b).abs().sum(axis=2)

    if metric == 'euclidean':
        if p < 1.:
            raise ValueError("too small p for p-norm")
        return cdist(a, b, 'minkovski', p=2.)

    if metric == 'minkovski':
        a = T.reshape(a, (a.shape[0], -1)).dimshuffle(0, 'x', 1)
        b = T.reshape(b, (b.shape[0], -1)).dimshuffle('x', 0, 1)
        return (a - b).norm(p, axis=2)

    elif metric == 'cosine':
        a = T.reshape(a, (a.shape[0], -1))
        a /= a.norm(2, axis=1).dimshuffle(0, 'x') + epsilon
        a = a.dimshuffle(0, 'x', 1)
        b = T.reshape(b, (b.shape[0], -1))
        b /= b.norm(2, axis=1).dimshuffle(0, 'x') + epsilon
        b = b.dimshuffle('x', 0, 1)
        return 1 - T.sum(a * b, axis=2)

    else:
        raise ValueError("invalid metric")


# Useful routines -----------------------------------------------------------------------

def compute_scores(predictions, targets, vocabulary):
    pred_cat = np.concatenate(predictions)
    labels_cat = np.concatenate(targets)

    jaccard_score = np.mean(
        [jaccard(onehot(l, vocabulary), onehot(p, vocabulary))
         for l, p in zip(targets, predictions)])
    framewise = np.mean(pred_cat == labels_cat)
    confusion = confusion_matrix(labels_cat, pred_cat).astype(np.double)

    return jaccard_score, framewise, confusion


# Custom NN layers ----------------------------------------------------------------------

class DurationMaskLayer(Layer):
    def __init__(self, incoming, max_time, batch_axis=0, **kwargs):
        super().__init__(incoming, **kwargs)
        self.batch_axis = batch_axis
        self.max_time = max_time

    def get_output_for(self, input, **kwargs):
        return T.lt(T.arange(self.max_time)[None, :], input[:, None])

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.max_time


def adjust_length(seq, size, axis=0, pad=0):
    # cut too long sequence
    seq = seq[[slice(None)] * axis + [slice(size)]]

    # pad too short sequence
    pad_shape = \
        seq.shape[:axis] \
        + (max(0, size - seq.shape[axis]),) \
        + seq.shape[axis + 1:]
    pad_value = np.full(pad_shape, pad, dtype=seq.dtype)
    seq = np.concatenate((seq, pad_value), axis=axis)

    return seq
