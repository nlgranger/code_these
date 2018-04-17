#!/bin/env python3

import os
import logging
import pickle as pkl

import numpy as np
from numpy.lib.format import open_memmap
from numpy.random import uniform
from scipy.ndimage.filters import gaussian_filter1d
import seqtools

from sltools.preprocess import interpolate_positions
from sltools.transform import Transformation, transform_durations, \
    transform_pose2d, transform_pose3d, transform_frames
from sltools.utils import split_seq, join_seqs

# from datasets import chalearn2014 as dataset
from datasets import devisign as dataset


cachedir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")
train_subset, val_subset, test_subset = None, None, None
pose2d_seqs = None
pose3d_seqs = None
frame_seqs = None
durations = None
labels = None
recordings = None
transformations = None
joints = dataset.JointType
vocabulary = np.arange(1, 21)

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
    global train_subset, val_subset, test_subset, labels, durations, \
        recordings, transformations, pose2d_seqs, pose3d_seqs

    # Create temporary directory
    if not os.path.exists(cachedir):
        os.mkdir(cachedir)

    # Load data

    # Chalearn
    # train_subset, val_subset, test_subset = dataset.default_splits()
    # pose2d_seqs = [dataset.positions(i) for i in range(len(dataset))]
    # pose3d_seqs = [dataset.positions_3d(i) for i in range(len(dataset))]
    # gloss_seqs = [dataset.glosses(i) for i in range(len(dataset))]
    #
    # # Eliminate strange gloss annotations
    # gloss_seqs_train = [dataset.glosses(r) for r in train_subset]
    # rejected = set()
    # for r, gseq in zip(train_subset, gloss_seqs_train):
    #     for i in range(len(gseq) - 1):
    #         if gseq[i + 1, 1] - gseq[i, 2] < 0:
    #             rejected.add(r)
    # train_subset = np.setdiff1d(train_subset, rejected)
    # if len(rejected) > 0:
    #     logging.warning("Eliminated sequences with invalid glosses: {}".format(rejected))
    #
    # # Interpolate missing poses and eliminate deteriorated training sequences
    # invalid_masks = smap(detect_invalid_pts, pose2d_seqs)
    # pose2d_seqs = smap(interpolate_positions, pose2d_seqs, invalid_masks)
    # pose3d_seqs = smap(interpolate_positions, pose3d_seqs, invalid_masks)
    #
    # rejected = np.where([np.mean(im[:, important_joints]) > .15
    #                      for im in invalid_masks])[0]
    # train_subset = np.setdiff1d(train_subset, rejected)
    # if len(rejected) > 0:
    #     logging.warning("eliminated {} sequences with missing positions"
    #                     .format(len(rejected)))
    #
    # # Extract isolated gestures
    # subseqs = [(r, g, start, stop)
    #            for r in range(len(dataset))
    #            for g, start, stop in dataset.glosses(r)]
    # labels = [g for r in range(len(dataset)) for g, _, _ in dataset.glosses(r)]
    # durations = [stop - start for _, _, start, stop in subseqs]
    # train_subset, val_subset, test_subset = (
    #     np.array([i for i, (r, g, _, _) in enumerate(subseqs)
    #               if (r in train_subset or r in val_subset) and g <= 10]),
    #     np.array([i for i, (r, g, _, _) in enumerate(subseqs)
    #               if (r in train_subset or r in val_subset) and g > 10]),
    #     np.array([]))
    # pose2d_seqs = [pose2d_seqs[seq][start:stop].astype(np.float32)
    #                for seq, _, start, stop in subseqs]
    # pose3d_seqs = [pose3d_seqs[seq][start:stop].astype(np.float32)
    #                for seq, _, start, stop in subseqs]

    # Devisign
    pose2d_seqs = [dataset.positions(i).astype(np.float32) for i in range(len(dataset))]
    pose3d_seqs = [dataset.positions_3d(i) for i in range(len(dataset))]
    durations = np.array([dataset.durations(r) for r in range(len(dataset))])
    labels = np.array([dataset.label(r) for r in range(len(dataset))])
    signers = np.array([dataset.subject(r) for r in range(len(dataset))])
    shuffled_labels = np.unique(labels)
    train_subset = np.where(np.isin(signers, [1, 2, 5, 6])
                            * np.isin(labels, shuffled_labels[:1000]))[0]
    val_subset = np.where(np.isin(signers, [3, 4, 7, 8])
                          * np.isin(labels, shuffled_labels[1000:]))[0]
    test_subset = np.array([], dtype=np.int64)

    pose2d_seqs = [dataset.positions(i).astype(np.float32)
                   for i in range(len(dataset))]
    pose3d_seqs = [dataset.positions_3d(i) for i in range(len(dataset))]
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
    ref2d = seqtools.smap(get_ref_pts, pose2d_seqs)
    ref3d = seqtools.smap(get_ref_pts, pose3d_seqs)

    zshifts = np.array([tgt_dist - rp[:, 2].mean() for rp in ref3d])

    recordings = np.arange(len(labels))
    transformations = [
        Transformation(ref2d=ref2d[r], ref3d=ref3d[r], zshift=zshifts[r])
        for r in recordings]

    # Precompute transformations for augmentation of the training set
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

    for _ in range(1):
        offset = len(transformations)
        recordings = np.concatenate([recordings, original_train_subset])
        transformations += [
            Transformation(ref2d=ref2d[r], ref3d=ref3d[r], flip_mapping=flip_mapping,
                           frame_width=640,
                           fliplr=uniform() < 0.15,
                           tilt=uniform(-7, 7) * np.pi / 180,
                           zshift=zshifts[r],
                           xscale=uniform(.85, 1.15), yscale=uniform(.85, 1.15),
                           zscale=uniform(.85, 1.15), tscale=uniform(.7, 1.3))
            for r in original_train_subset]
        train_subset = np.concatenate([train_subset,
                                       np.arange(offset, len(transformations))])

    # Precompute some augmentated values, save video transformation for later
    durations = np.array([transform_durations(durations[r], t)
                          for r, t in zip(recordings, transformations)])

    labels = np.array([labels[r] for r in recordings])

    pose2d_seqs = seqtools.gather(pose2d_seqs, recordings)
    pose2d_seqs = seqtools.smap(transform_pose2d, pose2d_seqs, transformations)

    pose3d_seqs = seqtools.gather(pose3d_seqs, recordings)
    pose3d_seqs = seqtools.smap(transform_pose3d, pose3d_seqs, transformations)

    # Export
    total_duration = sum(durations)
    subsequences = np.stack([np.cumsum(durations) - durations,
                             np.cumsum(durations)], axis=1)

    feat_size = pose2d_seqs[0].shape[1:]
    dump_file = os.path.join(cachedir, "pose2d_seqs.npy")
    storage = open_memmap(dump_file, 'w+', dtype=np.float32,
                          shape=(total_duration,) + feat_size)
    seqs_storage = split_seq(storage, subsequences)
    for i, s in enumerate(seqtools.prefetch(pose2d_seqs, 2, 20)):
        seqs_storage[i][...] = s

    feat_size = pose3d_seqs[0].shape[1:]
    dump_file = os.path.join(cachedir, "pose3d_seqs.npy")
    storage = open_memmap(dump_file, 'w+', dtype=np.float32,
                          shape=(total_duration,) + feat_size)
    seqs_storage = split_seq(storage, subsequences)
    for i, s in enumerate(seqtools.prefetch(pose3d_seqs, 2, 20)):
        seqs_storage[i][...] = s

    train_subset = train_subset.astype(np.int32)
    val_subset = val_subset.astype(np.int32)
    test_subset = test_subset.astype(np.int32)

    with open(os.path.join(cachedir, "data.pkl"), 'wb') as f:
        pkl.dump((durations, labels, recordings, transformations,
                  train_subset, val_subset, test_subset), f)

    print("done")


def reload():
    global train_subset, val_subset, test_subset, \
        durations, labels, recordings, transformations, \
        pose2d_seqs, pose3d_seqs, frame_seqs

    with open(os.path.join(cachedir, "data.pkl"), 'rb') as f:
        durations, labels, recordings, transformations, \
            train_subset, val_subset, test_subset = pkl.load(f)

    segments = np.stack([np.cumsum(durations) - durations,
                         np.cumsum(durations)], axis=1)

    pose2d_seqs = split_seq(
        np.load(os.path.join(cachedir, "pose2d_seqs.npy"), mmap_mode='r'),
        segments)

    pose3d_seqs = split_seq(
        np.load(os.path.join(cachedir, "pose3d_seqs.npy"), mmap_mode='r'),
        segments)

    frame_seqs = seqtools.smap(lambda r: np.array(dataset.bgr_frames(r)),
                               recordings)
    frame_seqs = seqtools.smap(transform_frames, frame_seqs, transformations)


if __name__ == "__main__":
    prepare()

else:
    reload()
