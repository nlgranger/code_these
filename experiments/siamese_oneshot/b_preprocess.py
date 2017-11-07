#!/bin/env python3

import os
from itertools import combinations
from time import time

import numpy as np
from numpy.lib.format import open_memmap
from scipy.ndimage.filters import gaussian_filter1d
from lproc import rmap

from experiments.siamese_oneshot.a_data import joints, tmpdir, train_subset, \
    durations, pose3d_seqs
from sltools.utils import split_seq, simple_mode_search


feat_seqs = None


def body_scale(positions, tgt_shoulder_w=.34):
    """Return apparent distance to the subject."""
    shoulder_w = simple_mode_search(
        np.linalg.norm(positions[:, joints.ShoulderLeft]
                       - positions[:, joints.ShoulderRight], axis=1))
    return tgt_shoulder_w / shoulder_w if shoulder_w > 0.2 else 1


def skel_feat(pose_seq):
    duration = pose_seq.shape[0]
    d = pose_seq.shape[-1]

    pose_seq = pose_seq * body_scale(pose_seq)

    pose_seq = gaussian_filter1d(
        pose_seq, sigma=1., axis=0, mode='nearest')

    points = [joints.HandLeft, joints.HandRight, joints.ElbowLeft, joints.ElbowRight,
              joints.ShoulderLeft, joints.ShoulderRight, joints.Head, joints.HipRight,
              joints.HipLeft]

    pairs = list(combinations(points, 2))

    angles = [(joints.ShoulderRight, joints.ElbowRight, joints.HandRight),
              (joints.ShoulderLeft, joints.ElbowLeft, joints.HandLeft),
              (joints.ShoulderLeft, joints.ShoulderRight, joints.ElbowRight),
              (joints.ElbowLeft, joints.ShoulderLeft, joints.ShoulderRight)]

    # Angle and orientation features
    cosin = np.empty((duration, len(angles)))
    crossin = np.empty((duration, len(angles) * (1 if d == 2 else 4)))

    for i, (j1, j2, j3) in enumerate(angles):
        v1 = pose_seq[:, j2, :] - pose_seq[:, j1, :]
        v1 /= np.linalg.norm(v1, axis=1, keepdims=True) + .001
        v2 = (pose_seq[:, j3, :] - pose_seq[:, j2, :])
        v2 /= np.linalg.norm(v2, axis=1, keepdims=True) + .001
        cosin[:, i] = np.sum(v1 * v2, axis=1)
        cross = np.cross(v1, v2)
        sinus = np.linalg.norm(v2, axis=1, keepdims=True) + .001
        if d == 3:
            crossin[:, i * 4:(i + 1) * 4] = np.concatenate([cross / sinus, sinus],
                                                           axis=1)
        else:
            crossin[:, i:i + 1] = sinus

    # Pairwise distances and position
    dists = np.stack(
        [np.linalg.norm(pose_seq[:, j1] - pose_seq[:, j2], axis=1)
         for j1, j2 in pairs], axis=1)

    diffs = np.concatenate(
        [(pose_seq[:, j1] - pose_seq[:, j2]).reshape(duration, -1)
         for j1, j2 in pairs], axis=-1)

    # Velocity / acceleration
    velocity = pose_seq[1:, points] - pose_seq[:-1, points]
    velocity = np.concatenate([
        np.zeros((1, len(points), pose_seq.shape[-1])),
        velocity]).reshape((duration, -1))

    jitters = pose_seq[2:, points] + pose_seq[:-2, points] - 2 * pose_seq[1:-1, points]
    jitters = np.concatenate(
        [np.zeros((1, len(points), pose_seq.shape[-1])),
         jitters,
         np.zeros((1, len(points), pose_seq.shape[-1]))]).reshape((duration, -1))

    # Positions
    ref = np.array([.5]) * (pose_seq[:, joints.ShoulderLeft]
                            + pose_seq[:, joints.ShoulderRight])
    smooth_ref = np.copy(ref)
    for i in range(1, len(smooth_ref)):
        smooth_ref[i] = 0.1 * smooth_ref[i] + .9 * smooth_ref[i - 1]
    pos = (pose_seq[:, points] - ref[:, None, :]).reshape(duration, -1)
    ref_diff = (ref[:, None, :] - smooth_ref[:, None, :]).reshape(duration, -1)

    # all feats
    feats = np.concatenate([cosin, crossin, velocity,
                            jitters, dists, diffs, pos, ref_diff], axis=1)

    # disable one feat
    # feats = np.concatenate([velocity,
    #                         jitters, dists, diffs, pos, ref_diff], axis=1)

    # temporal fusion
    # return np.concatenate([
    #     np.concatenate([np.repeat(feats[None, 0], 3, axis=0), feats[:-3]]),
    #     feats,
    #     np.concatenate([feats[3:], np.repeat(feats[None, 0], 3, axis=0)]),
    # ], axis=1)

    return feats


def prepare():
    # Processing pipeline
    feat_seqs = rmap(skel_feat, pose3d_seqs)

    # Export to file
    feat_size = feat_seqs[0][0].shape
    print("feat size: {}".format(feat_size))
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


def reload():
    global feat_seqs

    dump_file = os.path.join(tmpdir, 'skel_seqs.npy')
    storage = np.load(dump_file, mmap_mode='r')
    subsequences = np.stack([np.cumsum(durations) - durations,
                             np.cumsum(durations)], axis=1)
    feat_seqs = split_seq(storage, subsequences)


if __name__ == "__main__":
    prepare()

else:
    reload()
