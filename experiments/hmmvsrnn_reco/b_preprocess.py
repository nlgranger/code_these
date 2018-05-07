#!/bin/env python3

import os
import shelve
from datetime import timedelta
from time import time
from itertools import combinations

import numpy as np
import seqtools
from numpy.lib.format import open_memmap
from scipy.ndimage.filters import gaussian_filter1d
from sltools.utils import split_seq, simple_mode_search, crop
from sltools.preprocess import fix_contrast

from experiments.hmmvsrnn_reco.a_data import tmpdir, train_subset, \
    durations, pose2d_seqs, pose3d_seqs, frame_seqs, gloss_seqs, joints


# Exported values -----------------------------------------------------------------------

skel_feat_seqs = None
bgr_feat_seqs = None


# Settings ------------------------------------------------------------------------------

crop_size = (32, 32)


# ---------------------------------------------------------------------------------------

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
        v1 /= np.linalg.norm(v1, axis=1, keepdims=True) + .01
        v2 = (pose_seq[:, j3, :] - pose_seq[:, j2, :])
        v2 /= np.linalg.norm(v2, axis=1, keepdims=True) + .01
        cosin[:, i] = np.sum(v1 * v2, axis=1)
        cross = np.cross(v1, v2)
        sinus = np.linalg.norm(v2, axis=1, keepdims=True) + .01
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


def bgr_feats(frame_seq, pose2d_seq):
    d = len(pose2d_seq)
    cropw, croph = crop_size
    crop_seq = np.empty((d, 2, cropw + 30, croph + 30, 3), dtype=np.uint8)
    for c, f, p in zip(crop_seq, frame_seq, pose2d_seq):
        c[1] = crop(f, p[joints.WristLeft], (cropw + 30, croph + 30))
        c[0] = crop(f, p[joints.WristRight], (cropw + 30, croph + 30))

    output = np.empty((d, 2, cropw, croph), dtype=np.float32)
    output[:, 0] = fix_contrast(crop_seq[:, 0])[:, 15:-15, -15:15:-1]
    output[:, 1] = fix_contrast(crop_seq[:, 1])[:, 15:-15, 15:-15]

    return output


# ---------------------------------------------------------------------------------------

def transfer_feat_seqs(transfer_from, freeze_at):
    import theano
    import theano.tensor as T
    import lasagne
    from sltools.nn_utils import adjust_length
    from experiments.hmmvsrnn_reco.utils import reload_best_hmm, reload_best_rnn

    report = shelve.open(os.path.join(tmpdir, transfer_from))

    if report['meta']['modality'] == "skel":
        source_feat_seqs = [skel_feat_seqs]
    elif report['meta']['modality'] == "bgr":
        source_feat_seqs = [bgr_feat_seqs]
    elif report['meta']['modality'] == "fusion":
        source_feat_seqs = [skel_feat_seqs, bgr_feat_seqs]
    else:
        raise ValueError()

    # no computation required
    if freeze_at == "inputs":
        return source_feat_seqs

    # reuse cached features
    dump_file = os.path.join(tmpdir, report['meta']['experiment_name']
                             + "_" + freeze_at + "feats.npy")
    if os.path.exists(dump_file):
        boundaries = np.stack((np.cumsum(durations) - durations,
                               np.cumsum(durations)), axis=1)
        return [split_seq(np.load(dump_file, mmap_mode='r'), boundaries)]

    # reload model
    if report['meta']['model'] == "hmm":
        _, recognizer, _ = reload_best_hmm(report)
        l_in = recognizer.posterior.l_in
        if freeze_at == "embedding":
            l_feats = recognizer.posterior.l_feats
        elif freeze_at == "logits":
            l_feats = recognizer.posterior.l_raw
        elif freeze_at == "posteriors":
            l_feats = lasagne.layers.NonlinearityLayer(recognizer.posterior.l_out, T.exp)
        else:
            raise ValueError()
        batch_size, max_time, *_ = l_in[0].output_shape  # TODO: fragile
        warmup = recognizer.posterior.warmup

    else:
        _, model_dict, _ = reload_best_rnn(report)
        l_in = model_dict['l_in']
        l_feats = model_dict['l_feats']
        batch_size, max_time, *_ = l_in[0].output_shape  # TODO: fragile
        warmup = model_dict['warmup']

    feats_var = lasagne.layers.get_output(l_feats, deterministic=True)
    predict_batch_fn = theano.function([l.input_var for l in l_in], feats_var)

    step = max_time - 2 * warmup

    # turn sequences into chunks
    chunks = [(i, k, min(d, k + max_time))
              for i, d in enumerate(durations)
              for k in range(0, d - warmup, step)]
    chunked_sequences = []
    for feat in source_feat_seqs:
        def get_chunk(i, t1, t2, feat_=feat):
            return adjust_length(feat_[i][t1:t2], size=max_time, pad=0)

        chunked_sequences.append(seqtools.starmap(get_chunk, chunks))
    chunked_sequences = seqtools.collate(chunked_sequences)

    # turn into minibatches
    null_sample = chunked_sequences[0]
    n_features = len(null_sample)

    def collate(b):
        return [np.array([b[i][c] for i in range(batch_size)])
                for c in range(n_features)]

    minibatches = seqtools.batch(
        chunked_sequences, batch_size, pad=null_sample, collate_fn=collate)
    minibatches = seqtools.prefetch(minibatches, nworkers=2, max_buffered=10)

    # process
    batched_predictions = seqtools.starmap(predict_batch_fn, minibatches)
    batched_predictions = seqtools.add_cache(batched_predictions)
    chunked_predictions = seqtools.unbatch(batched_predictions, batch_size)

    # recompose
    feat_size = l_feats.output_shape[2:]
    storage = open_memmap(dump_file, 'w+', dtype=np.float32,
                          shape=(sum(durations),) + feat_size)
    subsequences = np.stack([np.cumsum(durations) - durations,
                             np.cumsum(durations)], axis=1)
    out_view = seqtools.split(storage, subsequences)

    for v, (s, start, stop) in zip(chunked_predictions, chunks):
        skip = warmup if start > 0 else 0
        out_view[s][start + skip:stop] = v[skip:stop - start]

    return [out_view]


def prepare():
    global feat_seqs

    # Processing pipeline for skeleton
    feat_seqs = seqtools.smap(skel_feat, pose3d_seqs)

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

    # Post processing (normalize over gesture poses)
    train_mask = np.full((total_duration,), False)
    for r in train_subset:
        sstart, _ = subsequences[r]
        for _, gstart, gstop in gloss_seqs[r]:
            train_mask[sstart + gstart:sstart + gstop] = True
    train_mask = np.random.permutation(np.where(train_mask)[0])[:100000]
    m = np.mean(storage[train_mask, :], axis=0, keepdims=True)
    s = np.std(storage[train_mask, :], axis=0, keepdims=True)
    storage[:, :] -= m
    storage[:, :] /= s
    # Modality-wise normalization (helps visualize)
    # for a, b in [(0, 4), (4, 20), (20, 47), (47, 74),
    #              (74, 110), (110, 218), (218, 245), (245, 248)]:
    #     m = np.mean(storage[train_mask, a:b], keepdims=True)
    #     s = np.std(storage[train_mask, a:b], keepdims=True)
    #     storage[:, a:b] -= m
    #     storage[:, a:b] /= s

    # Processing pipeline for BGR frames
    feat_seqs = seqtools.smap(bgr_feats, frame_seqs, pose2d_seqs)

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
    global skel_feat_seqs, bgr_feat_seqs

    skel_dump_file = os.path.join(tmpdir, 'skel_seqs.npy')
    bgr_dump_file = os.path.join(tmpdir, 'bgr_seqs.npy')
    skel_storage = np.load(skel_dump_file, mmap_mode='r')
    bgr_storage = np.load(bgr_dump_file, mmap_mode='r')
    subsequences = np.stack([np.cumsum(durations) - durations,
                             np.cumsum(durations)], axis=1)
    skel_feat_seqs = split_seq(skel_storage, subsequences)
    bgr_feat_seqs = split_seq(bgr_storage, subsequences)


if __name__ == "__main__":
    prepare()

else:
    reload()
