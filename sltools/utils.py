from typing import Sequence
import multiprocessing
import numpy as np
import cv2


# Math ----------------------------------------------------------------------------------

def simple_mode_search(x, n_iter=10):
    x = np.asarray(x)
    s = x.shape[1:]
    x = x.reshape(x.shape[0], -1)
    m = np.mean(x, axis=0)
    subset = np.full(len(x), True, dtype=np.bool)
    for _ in range(n_iter):
        m = np.mean(x[subset], axis=0)
        d = np.linalg.norm(x - m, axis=1)
        subset = d <= 2 * np.mean(d[subset])
        if subset.sum() == 0:
            break
    return m.reshape(s)


def dichotomy(test, a, b, it=5):
    m = (a + b) / 2.0
    if it == 0:
        return m
    elif test(m) > 0:
        return dichotomy(test, a, m, it - 1)
    else:
        return dichotomy(test, m, b, it - 1)


# list processing, indexing... ----------------------------------------------------------

class split_seq(Sequence):
    """Split a sequence into pieces."""

    def __init__(self, data, subseqs):
        self.data = data
        self.subseqs = subseqs

    def __len__(self):
        return len(self.subseqs)

    def __getitem__(self, item):
        start, stop = self.subseqs[item]
        return self.data[start:stop]


def join_seqs(sequences, output=None):
    durations = np.array([len(s) for s in sequences])
    boundaries = np.stack((np.cumsum(durations) - durations,
                           np.cumsum(durations)), axis=1)

    s0 = sequences[0]

    if output is None:
        output = np.empty((durations.sum(),) + s0[0].shape, dtype=s0.dtype)

    output[boundaries[0, 0]:boundaries[0, 1]] = s0
    for i in range(1, len(sequences)):
        output[boundaries[i, 0]:boundaries[i, 1]] = sequences[i]

    return output, boundaries


def break_long_seq(seq, maxlen, step=None):
    step = step or maxlen
    for i in range(0, len(seq), step):
        return seq[i:i + maxlen]


def gloss2seq(glosses, duration, default=-1, dtype=np.int32):
    seq = np.full((duration,), default, dtype=dtype)
    for l, start, stop in glosses:
        if start < 0 or stop > duration:
            raise ValueError
        seq[start:stop] = l
    return seq


def seq2gloss(seq):
    glosses = []

    current = seq[0]
    start = 0
    for i, v in enumerate(seq):
        if v != current:
            glosses.append((current, start, i))
            current = v
            start = i

    if start < len(seq):
        glosses.append((current, start, len(seq)))

    return glosses


# Multiprocessing helpers ---------------------------------------------------------------
# http://stackoverflow.com/a/16071616/5786475

def _parmap_worker(f, q_in, q_out):
    # Worker for `parmap`
    while True:
        i, x = q_in.get()
        if i is None:  # stop on sentinel value
            return
        q_out.put((i, f(*x)))


def parmap(func, *iterables, nprocs=0):
    """A multiprocessing-based equivalent of python's `map` which uses multiple threads
    to consume the iterables.

    :param func:
        A callable (lambdas or instance methods accepted).
    :param iterables:
        one or several iterables over arguments. If several iterables are given, _func_
        must take as many arguments as iterables, and arguments are extracted from all
        iterators in parallel.
    :param nprocs:
        Number of workers to use. 0 or negative values indicate the number of CPU
        cores to spare. Default: 0 (one worker by cpu core)
    """

    if nprocs <= 0:
        nprocs += multiprocessing.cpu_count()

    q_in = multiprocessing.Queue(2 * nprocs)
    q_out = multiprocessing.Queue(2 * nprocs)
    unordered_results = {}  # temporary storage for results

    proc = [multiprocessing.Process(target=_parmap_worker, 
                                    args=(func, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.start()

    iterables = zip(*iterables)
    n_buffered = 0
    n_done = 0

    for i, x in enumerate(iterables):
        if n_buffered >= 2 * nprocs:  # fetch some results after a while
            idx, v = q_out.get()
            unordered_results[idx] = v
            n_buffered -= 1

            # return them in correct order
            while n_done in unordered_results.keys():
                yield unordered_results.pop(n_done)
                n_done += 1

        # inject arguments
        q_in.put((i, x))
        n_buffered += 1

    while n_buffered > 0:  # extract remaining results
        idx, v = q_out.get()
        unordered_results[idx] = v
        n_buffered -= 1

        while n_done in unordered_results.keys():
            yield unordered_results.pop(n_done)
            n_done += 1

    for _ in range(nprocs):  # inject sentinel values to stop threads
        q_in.put((None, None))
    for p in proc:  # wait for threads to terminate
        p.join()


# Image ---------------------------------------------------------------------------------

def crop(img, pos, size) -> np.array:
    """Safely crop an image.

    :param img:
        input image
    :param pos:
        center coordinates (x, y) of the crop box, float values accepted
    :param size:
        crop box size (width, height)
    :return:
        cropped image of the closest area to pos that fits within the image
    """
    fh, fw = img.shape[:2]
    x, y = pos
    w, h = size

    # choose top-left point to fit box within img boundaries
    t = min(max(int(y - h / 2), 0), fh - h)
    l = min(max(int(x - w / 2), 0), fw - w)

    return img[t:t + h, l:l + w, ...]


def img_dist_normalize(img, cur_dist, tgt_dist, pad=0):
    h, w = img.shape[:2]
    factor = float(tgt_dist / cur_dist)

    if factor < 1:
        crop_h, crop_w = h - int(round(h * factor)), w - int(round(w * factor))
        t, l = crop_h // 2, crop_w // 2
        b, r = h - crop_h + t, w - crop_w + l
        return cv2.resize(img[t:b, l:r, ...], (w, h),
                          interpolation=cv2.INTER_AREA)

    else:
        small = cv2.resize(img, (0, 0), fx=1 / factor, fy=1 / factor)
        margin = h - small.shape[0], w - small.shape[1]
        l, t = margin[1] // 2, margin[0] // 2
        r, b = margin[1] - l, margin[0] - t
        return cv2.copyMakeBorder(small, t, b, l, r, cv2.BORDER_CONSTANT, value=pad)


def zmap_dist_normalize(zmap, cur_dist, tgt_dist, pad=4000):
    return img_dist_normalize(zmap, cur_dist, tgt_dist, pad=pad) \
        + np.array([tgt_dist - cur_dist], dtype=zmap.dtype)


def pose_dist_normalize(positions, frame_size, cur_dist, tgt_dist):
    w, h = frame_size
    factor = tgt_dist / cur_dist

    offset = np.array([w / 2, h / 2]) if positions.shape[-1] == 2 else np.zeros((3,))
    res = (positions - offset) / factor + offset
    return res


def pose3d_dist_normalize(positions, cur_dist, tgt_dist):
    factor = tgt_dist / cur_dist
    res = positions / factor
    res[:, :, 2] += (tgt_dist - cur_dist) / 1000
    return res
