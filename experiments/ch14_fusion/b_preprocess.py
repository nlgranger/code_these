#!/bin/env python3

import os
from datetime import timedelta
from time import time

import numpy as np
from lproc import rmap
from sltools.utils import split_seq
from numpy.lib.format import open_memmap

from experiments.ch14_fusion.a_data import tmpdir, train_subset, \
    durations, pose2d_seqs, pose3d_seqs, frame_seqs
from experiments.ch14_skel.b_preprocess import skel_feat
from experiments.ch14_bgr.b_preprocess import bgr_feats


feat_seqs = None


crop_size = (32, 32)


def prepare():
    global feat_seqs

    # Processing pipeline for skeleton
    feat_seqs = rmap(skel_feat, pose3d_seqs)

    # Export to file
    feat_size = feat_seqs[0][0].shape
    print("skel feat size: {}".format(feat_size))
    total_duration = sum(durations)
    subsequences = np.stack([np.cumsum(durations) - durations,
                             np.cumsum(durations)], axis=1)
    skel_dump_file = os.path.join(tmpdir, 'skel_seqs.npy')
    storage = open_memmap(skel_dump_file, 'w+', dtype=np.float32,
                          shape=(total_duration,) + feat_size)
    seqs_storage = split_seq(storage, subsequences)

    print("computing feats .... eta ..:..", end="", flush=True)
    t1 = time()
    for i, f in enumerate(feat_seqs):
        seqs_storage[i][...] = f
        eta = (time() - t1) / subsequences[i, 1] * (total_duration - subsequences[i, 1])
        print("\rcomputing feats {:3.0f}% eta {:02.0f}:{:02.0f}".format(
            subsequences[i, 1] / total_duration * 100, eta / 60, eta % 60),
            end="", flush=True)

    print("")

    # Post processing
    train_mask = np.full((total_duration,), False)
    for r in train_subset:
        start, stop = subsequences[r]
        train_mask[start:stop] = True
    train_mask = np.random.permutation(np.where(train_mask)[0])[:100000]
    m = np.mean(storage[train_mask, :], axis=0, keepdims=True)
    s = np.std(storage[train_mask, :], axis=0, keepdims=True) + 0.01
    storage -= m
    storage /= s

    # Processing pipeline for BGR frames
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

    skel_dump_file = os.path.join(tmpdir, 'skel_seqs.npy')
    bgr_dump_file = os.path.join(tmpdir, 'bgr_seqs.npy')
    skel_storage = np.load(skel_dump_file, mmap_mode='r')
    bgr_storage = np.load(bgr_dump_file, mmap_mode='r')
    subsequences = np.stack([np.cumsum(durations) - durations,
                             np.cumsum(durations)], axis=1)
    skel_feat_seqs = split_seq(skel_storage, subsequences)
    bgr_feat_seqs = split_seq(bgr_storage, subsequences)
    feat_seqs = rmap(lambda x1, x2: (x1, x2), skel_feat_seqs, bgr_feat_seqs)


if __name__ == "__main__":
    prepare()

else:
    reload()
