#!/bin/env python3

import cv2
import os
import sys
import re
from enum import IntEnum
import numpy as np
import logging


class Joints(IntEnum):
    Nose = 0,
    Neck = 1,
    RShoulder = 2,
    RElbow = 3,
    RWrist = 4,
    LShoulder = 5,
    LElbow = 6,
    LWrist = 7,
    RHip = 8,
    RKnee = 9,
    RAnkle = 10,
    LHip = 11,
    LKnee = 12,
    LAnkle = 13,
    REye = 14,
    LEye = 15,
    REar = 16,
    LEar = 17


important_joints = [Joints.LWrist, Joints.LElbow, Joints.LShoulder,
                    Joints.RShoulder, Joints.RElbow, Joints.RWrist]


def process_recording(split, batch, rec, datadir, glosses_str):
    color_file = os.path.join(datadir, "videos", split, batch, rec + ".M.avi")
    duration = int(cv2.VideoCapture(color_file).get(cv2.CAP_PROP_FRAME_COUNT))
    assert duration > 0
    fps = cv2.VideoCapture(color_file).get(cv2.CAP_PROP_FPS)

    depth_file = os.path.join(datadir, "videos", split, batch, rec + ".K.avi")
    assert int(cv2.VideoCapture(depth_file).get(cv2.CAP_PROP_FRAME_COUNT)) == duration

    poses = np.zeros((duration, 18, 2), dtype=np.int32)
    pose_file = os.path.join(datadir, "poses", split, batch, rec + ".npy")
    pose_data = np.load(pose_file)
    time = pose_data[:, 0].astype(np.int) - 1
    raw_poses = pose_data[:, 1:].reshape((-1, 18, 3))
    confidence = np.zeros((duration,))
    for t, p in zip(time, raw_poses):
        if t < 0 or t >= duration:
            logging.warning(
                "outatime {} {}/{}".format((split, batch, rec), t, duration))
            continue
        c = np.sum(p[important_joints, 2])
        if c > confidence[t]:
            poses[t] = p[:, :2]
            confidence[t] = c

    glosses = []
    for gloss_str in glosses_str:
        m = re.fullmatch(r"([0-9]+),([0-9]+):([0-9]+)", gloss_str)
        assert m is not None
        start, stop, gloss = m.groups()
        glosses.append((int(gloss), int(start) - 1, int(stop)))

    return duration, fps, poses, glosses


def main():
    assert len(sys.argv) == 2
    datadir = sys.argv[-1]

    # Load annotations
    raw_glosses = {}
    for split in ["train", "valid", "test"]:
        raw_glosses[split] = {}
        with open(os.path.join(datadir, "annotations", split + ".txt")) as f:
            for line in f:
                recording, *rec_glosses = line.split()
                raw_glosses[os.path.join(split, recording)] = rec_glosses

    # Process recordings data
    metadata = []
    poses = {}
    glosses = {}
    metadata_dtype = [('id', np.unicode_, 17), ('duration', 'u4'),
                      ('fps', 'f2'), ('skel_data_off', 'u8'), ('lbl_data_off', 'u8'),
                      ('n_labels', 'u4')]
    for split in ["train", "valid", "test"]:
        for batch in os.listdir(os.path.join(datadir, "videos", split)):
            for file in os.listdir(os.path.join(datadir, "videos", split, batch)):
                if file.endswith(".M.avi"):
                    rec = file[:-6]
                    duration, fps, rec_poses, rec_glosses = process_recording(
                        split, batch, rec, datadir,
                        raw_glosses[os.path.join(split, batch, rec)])
                    rec_id = os.path.join(split, batch, rec)
                    metadata.append(np.array(
                        [(rec_id, duration, fps, 0, 0, len(rec_glosses))],
                        dtype=metadata_dtype))
                    poses[rec_id] = rec_poses
                    glosses[rec_id] = rec_glosses

    # Concatenate and export
    metadata = np.concatenate(metadata)
    metadata.sort(order='id')
    metadata['skel_data_off'] = np.cumsum(metadata['duration']) - metadata['duration']
    metadata['lbl_data_off'] = np.cumsum(metadata['n_labels']) - metadata['n_labels']
    poses = np.concatenate([poses[r] for r in metadata['id']])
    glosses = np.concatenate([glosses[r] for r in metadata['id'] if len(glosses[r]) > 0])

    np.save(os.path.join(datadir, 'rec_info.npy'), metadata)
    np.save(os.path.join(datadir, 'poses.npy'), poses)
    np.save(os.path.join(datadir, 'glosses.npy'), glosses)


if __name__ == "__main__":
    main()
