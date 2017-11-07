#!/bin/env python3

import os
from time import time
from datetime import timedelta

import numpy as np
from lproc import rmap
from sltools.utils import crop, split_seq
from numpy.lib.format import open_memmap

from experiments.ch14_bgr.a_data import tmpdir, joints, \
    durations, pose2d_seqs, frame_seqs


feat_seqs = None


crop_size = (32, 32)


def bgr_feats(frame_seq, pose2d_seq):
    return np.array([
        [crop(f, p[joints.WristLeft], crop_size)[:, ::-1],
         crop(f, p[joints.WristRight], crop_size)]
        for f, p in zip(frame_seq, pose2d_seq)])


def prepare():
    global feat_seqs

    # Processing pipeline
    feat_seqs = rmap(bgr_feats, frame_seqs, pose2d_seqs)

    # Export to file
    feat_size = feat_seqs[0][0].shape
    print("feat size: {}".format(feat_size))
    total_duration = sum(durations)
    subsequences = np.stack([np.cumsum(durations) - durations,
                             np.cumsum(durations)], axis=1)
    bgr_dump_file = os.path.join(tmpdir, 'bgr_seqs.npy')
    storage = open_memmap(bgr_dump_file, 'w+', dtype=np.float32,
                          shape=(total_duration,) + feat_size)
    seqs_storage = split_seq(storage, subsequences)

    print("computing feats .... eta ..:..:..", end="", flush=True)
    t1 = time()
    for i, f in enumerate(feat_seqs):
        seqs_storage[i][...] = f
        eta = (time() - t1) / subsequences[i, 1] * (total_duration - subsequences[i, 1])
        print("\rcomputing feats {:3.0f}% eta {}".format(
            subsequences[i, 1] / total_duration * 100, timedelta(seconds=eta)),
            end="", flush=True)

    print("")


def reload():
    global feat_seqs

    dump_file = os.path.join(tmpdir, 'bgr_seqs.npy')
    storage = np.load(dump_file, mmap_mode='r')
    subsequences = np.stack([np.cumsum(durations) - durations,
                             np.cumsum(durations)], axis=1)
    feat_seqs = split_seq(storage, subsequences)


if __name__ == "__main__":
    prepare()

else:
    reload()
