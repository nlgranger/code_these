#!/bin/env python3

import logging
import os
import pickle as pkl

import numpy as np
from lproc import rmap
from numpy.random import uniform
from scipy.ndimage.filters import gaussian_filter1d

from datasets import chalearn2014 as dataset
from sltools.preprocess import interpolate_positions
from sltools.transform import Transformation, \
    transform_durations, transform_glosses, transform_pose3d
from sltools.utils import split_seq, join_seqs

tmpdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")
gloss_seqs = None
train_subset, val_subset, test_subset = None, None, None
pose3d_seqs = None
frame_seqs = None
durations = None
joints = dataset.JointType

tgt_dist = 4  # desired apparent distance (in meters?) from camera to subject
important_joints = np.isin(np.arange(dataset.positions(0).shape[1]),
                           [joints.WristLeft, joints.ElbowLeft, joints.ShoulderLeft,
                            joints.WristRight, joints.ElbowRight, joints.ShoulderRight])


def detect_invalid_pts(pose_seq):
    return ((pose_seq[:, :, 0] <= 0)
            + (pose_seq[:, :, 1] <= 0)
            + (pose_seq[:, :, 0] >= 479)
            + (pose_seq[:, :, 1] >= 639)) * important_joints[None, :]


def get_ref_pts(pose_seq):
    out = pose_seq[:, [joints.ShoulderRight, joints.ShoulderLeft], :].mean(axis=1)
    out = gaussian_filter1d(out, sigma=3, axis=0, mode='nearest').astype(pose_seq.dtype)
    return out


def prepare():
    global train_subset, val_subset, test_subset, durations, gloss_seqs, pose3d_seqs

    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)

    augmentation_factor = 5

    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)

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
    invalid_masks = rmap(detect_invalid_pts, pose2d_seqs)
    pose3d_seqs = rmap(interpolate_positions, pose3d_seqs, invalid_masks)

    rejected = np.where([np.mean(im[:, important_joints]) > .15
                         for im in invalid_masks])[0]
    train_subset = np.setdiff1d(train_subset, rejected)
    if len(rejected) > 0:
        logging.warning("Eliminated {} sequences with missing positions"
                        .format(len(rejected)))

    # Default preprocessing
    ref3d = rmap(get_ref_pts, pose3d_seqs)

    zshifts = np.array([tgt_dist - rp[:, 2].mean() for rp in ref3d])

    transformations = [
        (r, Transformation(ref3d=ref3d[r], zshift=zshifts[r]))
        for r in np.arange(len(dataset))]

    # Augment training set
    original_train_subset = train_subset
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

    for _ in range(augmentation_factor - 1):
        offset = len(transformations)
        transformations += [
            (r, Transformation(ref3d=ref3d[r], flip_mapping=flip_mapping,
                               frame_width=dataset.bgr_frames(0)[0].shape[1],
                               fliplr=uniform() < 0.15,
                               tilt=uniform(-7, 7) * np.pi / 180,
                               zshift=zshifts[r],
                               xscale=uniform(.85, 1.15), yscale=uniform(.85, 1.15),
                               zscale=uniform(.85, 1.15), tscale=uniform(.7, 1.3)))
            for r in original_train_subset]
        train_subset = np.concatenate([train_subset,
                                       np.arange(offset, len(transformations))])

    durations = np.array([transform_durations(dataset.durations(r), t)
                          for r, t in transformations])

    gloss_seqs = [transform_glosses(dataset.glosses(r), dataset.durations(r), t)
                  for r, t in transformations]

    pose3d_seqs = [transform_pose3d(pose3d_seqs[r], t) for r, t in transformations]

    # Export
    np.save(os.path.join(tmpdir, "pose3d_seqs.npy"), join_seqs(pose3d_seqs)[0])

    with open(os.path.join(tmpdir, "data.pkl"), 'wb') as f:
        pkl.dump((durations, gloss_seqs, transformations,
                  train_subset, val_subset, test_subset), f)


def reload():
    global train_subset, val_subset, test_subset, \
        durations, gloss_seqs, pose3d_seqs

    with open(os.path.join(tmpdir, "data.pkl"), 'rb') as f:
        durations, gloss_seqs, transformations, \
            train_subset, val_subset, test_subset = pkl.load(f)

    segments = np.stack((np.cumsum(durations) - durations,
                         np.cumsum(durations)), axis=1)

    pose3d_seqs = split_seq(
        np.load(os.path.join(tmpdir, "pose3d_seqs.npy"), mmap_mode='r'),
        segments)


if __name__ == "__main__":
    prepare()

else:
    reload()