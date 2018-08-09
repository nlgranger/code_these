#!/bin/env python3

import os
import logging
import pickle as pkl
from functools import partial

import numpy as np
from numpy.random import uniform
import seqtools

from sltools.preprocess import interpolate_positions
from sltools.transform import \
    TransformationType, transform_durations, transform_glosses, \
    transform_pose2d, transform_pose3d, transform_frames

from datasets import chalearn2014 as dataset


cachedir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache.run7")
gloss_seqs = None
train_subset, val_subset, test_subset = None, None, None
pose2d_seqs = None
pose3d_seqs = None
frame_seqs = None
durations = None
joints = dataset.JointType
vocabulary = np.arange(1, 21)

tgt_dist = 4  # desired apparent distance (in meters?) from camera to subject
important_joints = np.array([joints.WristLeft, joints.ElbowLeft, joints.ShoulderLeft,
                             joints.WristRight, joints.ElbowRight, joints.ShoulderRight])
flip_mapping = ([joints.ShoulderRight, joints.ElbowRight,
                 joints.WristRight, joints.HandRight, joints.ShoulderLeft,
                 joints.ElbowLeft, joints.WristLeft, joints.HandLeft,
                 joints.HipRight, joints.KneeRight, joints.AnkleRight,
                 joints.FootRight, joints.HipLeft, joints.KneeLeft,
                 joints.AnkleLeft, joints.FootLeft],
                [joints.ShoulderLeft, joints.ElbowLeft,
                 joints.WristLeft, joints.HandLeft, joints.ShoulderRight,
                 joints.ElbowRight, joints.WristRight, joints.HandRight,
                 joints.HipLeft, joints.KneeLeft, joints.AnkleLeft,
                 joints.FootLeft, joints.HipRight, joints.KneeRight,
                 joints.AnkleRight, joints.FootRight])


def detect_invalid_pts(pose_seq):
    mask = np.isin(np.arange(pose_seq.shape[1]), important_joints)[None, :]
    return ((pose_seq[:, :, 0] <= 0)
            + (pose_seq[:, :, 1] <= 0)
            + (pose_seq[:, :, 0] >= 479)
            + (pose_seq[:, :, 1] >= 639)) & mask


def get_ref_pts(pose_seq):
    return np.mean(np.mean(
        pose_seq[:, [joints.ShoulderRight, joints.ShoulderLeft], :],
        axis=1), axis=0)


def prepare():
    global train_subset, val_subset, test_subset, \
        durations, gloss_seqs, pose2d_seqs, pose3d_seqs

    # Create temporary directory
    if not os.path.exists(cachedir):
        os.mkdir(cachedir)

    # Load data
    train_subset, val_subset, test_subset = dataset.default_splits()
    pose2d_seqs = [dataset.positions(i) for i in range(len(dataset))]
    pose3d_seqs = [dataset.positions_3d(i) for i in range(len(dataset))]

    # Eliminate strange gloss annotations
    gloss_seqs_train = [dataset.glosses(r) for r in train_subset]
    rejected = set()
    for r, gseq in zip(train_subset, gloss_seqs_train):
        for i in range(len(gseq) - 1):
            if gseq[i + 1, 1] - gseq[i, 2] < 0:
                rejected.add(r)
    train_subset = np.setdiff1d(train_subset, rejected)
    if len(rejected) > 0:
        logging.warning("Eliminated sequences with invalid glosses: {}".format(rejected))

    # Interpolate missing poses and eliminate deteriorated training sequences
    invalid_masks = seqtools.smap(detect_invalid_pts, pose2d_seqs)
    pose2d_seqs = seqtools.smap(interpolate_positions, pose2d_seqs, invalid_masks)
    pose3d_seqs = seqtools.smap(interpolate_positions, pose3d_seqs, invalid_masks)

    rejected = np.where([np.mean(im[:, important_joints]) > .15
                         for im in invalid_masks])[0]
    train_subset = np.setdiff1d(train_subset, rejected)
    if len(rejected) > 0:
        logging.warning("eliminated {} sequences with missing positions"
                        .format(len(rejected)))

    # Default preprocessing
    ref2d = seqtools.add_cache(seqtools.smap(get_ref_pts, pose2d_seqs), cache_size=1)
    ref3d = seqtools.add_cache(seqtools.smap(get_ref_pts, pose3d_seqs), cache_size=1)

    transformations = np.rec.array(
        [(r2, r3, False, 0, tgt_dist - r3[2], 1., 1., 1., 1.)
         for r2, r3 in zip(ref2d, ref3d)],
        dtype=TransformationType)

    # Precompute transformations for augmentation of the training set
    original_train_subset = train_subset

    rec_mapping = np.arange(len(dataset))
    for _ in range(5 - 1):
        offset = len(rec_mapping)
        new_subset = np.arange(offset, offset + len(original_train_subset))

        newt = np.repeat(transformations[0], len(new_subset), axis=0).view(np.recarray)
        newt.fliplr = uniform(size=len(newt)) < 0.15
        newt.tilt += uniform(-7, 7, size=len(newt)) * np.pi / 180
        newt.xscale += uniform(.85, 1.15, size=len(newt))
        newt.yscale += uniform(.85, 1.15, size=len(newt))
        newt.zscale += uniform(.85, 1.15, size=len(newt))
        newt.tscale += uniform(.85, 1.15, size=len(newt))

        rec_mapping = np.concatenate([rec_mapping, original_train_subset])
        transformations = np.concatenate([transformations, newt]).view(np.recarray)
        train_subset = np.concatenate([train_subset, new_subset])

    # Apply transformations (if they are cheap to compute)
    durations = np.array([transform_durations(dataset.durations(r), t)
                          for r, t in zip(rec_mapping, transformations)])

    gloss_seqs = [transform_glosses(dataset.glosses(r), dataset.durations(r), t)
                  for r, t in zip(rec_mapping, transformations)]

    pose2d_seqs = seqtools.gather(pose2d_seqs, rec_mapping)
    pose2d_seqs = seqtools.smap(
        partial(transform_pose2d, flip_mapping=flip_mapping, frame_width=640),
        pose2d_seqs, transformations)

    pose3d_seqs = seqtools.gather(pose3d_seqs, rec_mapping)
    pose3d_seqs = seqtools.smap(
        partial(transform_pose3d, flip_mapping=flip_mapping),
        pose3d_seqs, transformations)

    # Export
    np.save(os.path.join(cachedir, "pose2d_seqs.npy"), seqtools.concatenate(pose2d_seqs))
    np.save(os.path.join(cachedir, "pose3d_seqs.npy"), seqtools.concatenate(pose3d_seqs))

    with open(os.path.join(cachedir, "data.pkl"), 'wb') as f:
        pkl.dump((durations, gloss_seqs, rec_mapping, transformations,
                  train_subset, val_subset, test_subset), f)


def reload():
    global train_subset, val_subset, test_subset, \
        durations, gloss_seqs, pose2d_seqs, pose3d_seqs, frame_seqs

    with open(os.path.join(cachedir, "data.pkl"), 'rb') as f:
        durations, gloss_seqs, rec_mapping, transformations, \
            train_subset, val_subset, test_subset = pkl.load(f)

    segments = np.stack([np.cumsum(durations) - durations,
                         np.cumsum(durations)], axis=1)

    pose2d_seqs = seqtools.split(
        np.load(os.path.join(cachedir, "pose2d_seqs.npy"), mmap_mode='r'),
        segments)

    pose3d_seqs = seqtools.split(
        np.load(os.path.join(cachedir, "pose3d_seqs.npy"), mmap_mode='r'),
        segments)

    frame_seqs = seqtools.smap(lambda r: dataset.bgr_frames(r), rec_mapping)
    frame_seqs = seqtools.smap(transform_frames, frame_seqs, transformations)


if __name__ == "__main__":
    prepare()

else:
    reload()
