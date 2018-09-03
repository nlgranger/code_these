import sys
import multiprocessing
import queue
from multiprocessing.sharedctypes import RawArray
from weakref import finalize
import traceback
import logging
import numpy as np
from theano import tensor as T
import lasagne
from lasagne.layers import Layer
from sklearn.metrics import confusion_matrix
import tblib
from seqtools.evaluation import AsyncSequence, JobStatus
from seqtools.utils import SharedCtypeQueue
from memory_profiler import profile
import signal


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


def softmax(x, axis=-1):
    return T.exp(log_softmax(x, axis=axis))


def categorical_crossentropy_logdomain(log_predictions, targets):
    """Compute categorical cross entropy given predictions in log-domain along last axis.
    """
    data_shape = targets.shape
    print()
    log_predictions = T.reshape(log_predictions, (-1, log_predictions.shape[-1]))
    targets = T.flatten(targets)
    return T.reshape(-log_predictions[T.arange(targets.shape[0]), targets],
                     data_shape)


def multiclass_hinge_loss(predictions, targets, delta):
    if targets.ndim == predictions.ndim:
        targets = T.argmax(targets, axis=-1)
    elif targets.ndim + 1 != predictions.ndim:
        raise ValueError("predictions and target dimensions mismatch")

    *original_shape, num_cls = predictions.shape
    predictions = T.reshape(predictions, (-1, num_cls))
    targets = T.flatten(targets)
    num_preds = targets.shape[0]

    pred_target = predictions[T.arange(num_preds), targets]
    predictions = T.set_subtensor(predictions[T.arange(num_preds), targets],
                                  T.min(predictions) - 1.)
    pred_closest = T.max(predictions, axis=1)

    return T.reshape(T.nnet.relu(pred_closest - pred_target + delta), original_shape)


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


# Fast Minibatch loader -----------------------------------------------------------------

class MultiprocessSequence(AsyncSequence):
    # AsyncSequence provides essential boilerplate code, particularly
    # locking and race condition management.
    # We then adapt the code from seqtools.evaluation.MultiprocessSequence
    # to specialize on numpy objects.
    def __init__(self, sequence, max_cached,
                 nworkers=0, timeout=1, start_hook=None, anticipate=None):
        """Wraps a sequence to use workers that prefetch values ahead.

        Args:
            sequence:
                A sequence containing tuples of numpy arrays, for
                example (value, label) pairs. The arrays must have
                the same type and shape across items.
            max_cached:
                Maximum number of locally cached precomputed values.
            nworkers:
                Number of prefetching workers
            timeout:
                Maximum idle time for active workers.
            start_hook:
                User specified callable run by each worker after start
                (for example random.seed).
            anticipate:
                Callable function taking an item index and returning the
                index that is the most likely to be accessed after that.
                Defaults to monotonic access.

        .. warning::

            The returned values are _views on buffers_, anytime a new item is requested,
            the values of previously returned items should be considered compromised.
        """
        if nworkers <= 0:
            nworkers = max(1, multiprocessing.cpu_count() - nworkers)
        if len(sequence) == 0:
            raise ValueError("empty sequences not supported")

        q_in = SharedCtypeQueue("Ll", maxsize=max_cached)
        q_out = SharedCtypeQueue("LLb", maxsize=max_cached)

        super(MultiprocessSequence, self).__init__(
            max_cached, q_in, q_out, timeout, anticipate)

        self.sequence = sequence
        self.nworkers = nworkers
        self.start_hook = start_hook

        # setup cache in shared memory
        sample = self.sequence[0]
        self.buffers = []
        for field in sample:
            self.buffers.append((
                RawArray(memoryview(field).format, max_cached * field.size),
                field.dtype, field.shape))
        self.values_cache = []
        for i in range(max_cached):
            slot = []
            for b, t, s in self.buffers:
                field_buffer = np.reshape(np.reshape(
                    np.frombuffer(b, t), (-1,) + s)[i:i + 1], s)
                field_buffer.flags.writeable = False
                slot.append(field_buffer)
            self.values_cache.append(tuple(slot))
        manager = multiprocessing.Manager()
        self.errors_cache = manager.list([None, ""] * max_cached)

        # setup workers
        self.workers = []
        finalize(self, MultiprocessSequence._finalize, self)  # ensure proper termination
        self.start_workers()

    def __len__(self):
        return len(self.sequence)

    def start_workers(self):
        for i, w in enumerate(self.workers):
            if not w.is_alive():
                w.join()
                self.workers[i] = multiprocessing.Process(
                    target=self.__class__.target,
                    args=(self.sequence, self.buffers, self.errors_cache,
                          self.q_in, self.q_out,
                          self.timeout, self.start_hook))
                self.workers[i].start()
        for _ in range(len(self.workers), self.nworkers):
            self.workers.append(multiprocessing.Process(
                target=self.__class__.target,
                args=(self.sequence, self.buffers, self.errors_cache,
                      self.q_in, self.q_out,
                      self.timeout, self.start_hook)))
            self.workers[-1].start()

    def read_cache(self, slot):
        return self.values_cache[slot]

    def reraise_failed_job(self, slot):
        error, trace = self.errors_cache[slot]
        self.errors_cache[slot] = (None, "")
        if error is not None:
            raise error.with_traceback(trace.as_traceback())
        else:
            raise RuntimeError(trace)

    @staticmethod
    @profile
    def target(sequence, buffers, errors_cache, q_in, q_out,
               timeout, start_hook):
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # let parent handle signals

        if start_hook is not None:
            start_hook()

        buffers = tuple(np.reshape(np.frombuffer(b, t), (-1,) + s)
                        for b, t, s in buffers)

        try:
            while True:
                try:
                    item, slot = q_in.get(timeout=timeout)
                except queue.Empty:
                    return
                if slot < 0:
                    return

                try:
                    for field, buffer in zip(sequence[item], buffers):
                        buffer[slot] = field
                except Exception as error:
                    _, _, trace = sys.exc_info()
                    trace = tblib.Traceback(trace)
                    tb_str = traceback.format_exc(20)
                    # noinspection PyBroadException
                    try:  # this one may fail if ev is not picklable
                        errors_cache[slot] = error, trace
                    except Exception:
                        errors_cache[slot] = None, tb_str

                    q_out.put_nowait((item, slot, JobStatus.FAILED))

                else:
                    q_out.put_nowait((item, slot, JobStatus.DONE))

        except IOError:
            return  # parent probably died

    @staticmethod
    def _finalize(obj):
        # drain input queue
        while not obj.q_in.empty():
            try:
                obj.q_in.get_nowait()
            except queue.Empty:
                pass

        # send termination signals
        for _ in obj.workers:
            obj.q_in.put((0, -1))

        for worker in obj.workers:
            worker.join()
