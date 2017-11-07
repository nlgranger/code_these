import cv2
import os
from zipfile import ZipFile
import csv
import argparse
import numpy as np


def process_recording(datadir, rec_num):
    color_file = os.path.join(datadir, 'Sample{:04d}_color.mp4'.format(rec_num))
    duration = int(cv2.VideoCapture(color_file).get(cv2.CAP_PROP_FRAME_COUNT))
    assert duration > 0
    fps = cv2.VideoCapture(color_file).get(cv2.CAP_PROP_FPS)

    depth_file = os.path.join(datadir, 'Sample{:04d}_depth.mp4'.format(rec_num))
    assert int(cv2.VideoCapture(depth_file).get(cv2.CAP_PROP_FRAME_COUNT)) == duration

    skel_file = os.path.join(datadir, 'Sample{:04d}_skeleton.csv'.format(rec_num))
    with open(skel_file, 'r', newline='') as csvfile:
        skelreader = csv.reader(csvfile, delimiter=',')
        skeldata = np.array([[float(v) for v in row] for row in skelreader],
                            dtype=np.float32)
        skeldata = skeldata.reshape(-1, 20, 9)
        assert duration == len(skeldata)

    data_file = os.path.join(datadir, 'Sample{:04d}_data.csv'.format(rec_num))
    with open(data_file, 'r', newline='') as csvfile:
        duration_, fps_, max_depth = csvfile.readline().split(',')
        duration_, fps_, max_depth = int(duration_), float(fps_), int(max_depth)
        assert duration_ == duration and -1e-5 < (fps - fps_) < 1e-5

    lbl_file = os.path.join(datadir, 'Sample{:04d}_labels.csv'.format(rec_num))
    with open(lbl_file, 'r', newline='') as csvfile:
        labelreader = csv.reader(csvfile, delimiter=',')
        labels = np.array([[int(v) for v in row] for row in labelreader])
        labels[:, 1:] -= 1  # 0 based indexing, TODO: decrease stop as well?
        labels = labels[np.lexsort([labels[:, 1], labels[:, 2]])]
        duplicates = []
        for i in range(len(labels) - 1):
            if np.all(labels[i] == labels[i + 1]):
                duplicates.append(i)
        if len(duplicates) > 0:
            labels = np.delete(labels, duplicates, axis=0)
            print("deleted {} duplicate(s) from Sample{:04d}".format(
                len(duplicates), rec_num))

    return duration, fps, skeldata, max_depth, labels


def main():
    parser = argparse.ArgumentParser(
        description='Compile Montalbano v2 metadata and annotations into binary files')
    parser.add_argument('archive_dir', metavar='ARCHIVEDIR', type=str,
                        help='path to data archives')
    parser.add_argument('data_dir', metavar='DATADIR', type=str,
                        help='path to the recordings/annotations/labels files'
                             ' (all extracted flat out inside without subdirs)')
    parser.add_argument('dest_dir', metavar='DESTDIR', type=str,
                        help='where the resulting files should go')
    args = parser.parse_args()

    splits = {}
    metadata = []
    labels = {}
    skel_data = {}

    # Identify training, validation and testing splits

    for a in ["Train1.zip", "Train2.zip", "Train3.zip", "Train4.zip", "Train5.zip"]:
        with ZipFile(os.path.join(args.archive_dir, a), 'r') as ar_zip:
            for recording in ar_zip.infolist():
                splits[int(recording.filename[-8:-4])] = 0

    for a in ["Validation1.zip", "Validation2.zip", "Validation3.zip"]:
        with ZipFile(os.path.join(args.archive_dir, a), 'r') as ar_zip:
            for recording in ar_zip.infolist():
                splits[int(recording.filename[-8:-4])] = 1

    for a in ["Test1e.zip", "Test2e.zip", "Test3e.zip", "Test4e.zip", "Test5e.zip"]:
        with ZipFile(os.path.join(args.archive_dir, a), 'r') as ar_zip:
            for recording in ar_zip.infolist():
                splits[int(recording.filename[-8:-4])] = 2

    # Process data

    for rec_num, split in splits.items():
        duration, fps, rec_skel_data, max_depth, rec_ann = \
            process_recording(args.data_dir, rec_num)
        metadata.append(np.array(
            [(rec_num, split, duration, fps, max_depth, 0, 0, len(rec_ann))],
            dtype=[('num', 'u2'), ('split', 'u1'), ('duration', 'u4'), ('fps', 'f2'),
                   ('max_depth', 'u2'), ('skel_data_off', 'u8'), ('lbl_data_off', 'u8'),
                   ('n_labels', 'u2')]))
        skel_data[rec_num] = rec_skel_data
        labels[rec_num] = rec_ann

    # Compile and export

    metadata = np.concatenate(metadata)
    metadata.sort(order='num')
    metadata['skel_data_off'] = np.cumsum(metadata['duration']) - metadata['duration']
    metadata['lbl_data_off'] = np.cumsum(metadata['n_labels']) - metadata['n_labels']
    skel_data = np.concatenate([skel_data[r] for r in metadata['num']])
    labels = np.concatenate([labels[r] for r in metadata['num']])

    np.save(os.path.join(args.dest_dir, 'rec_info.npy'), metadata)
    np.save(os.path.join(args.dest_dir, 'skel_data.npy'), skel_data)
    np.save(os.path.join(args.dest_dir, 'labels.npy'), labels)


if __name__ == "__main__":
    main()
