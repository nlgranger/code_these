#!/bin/env python3

import os
import sys
import numpy as np
from numpy.lib.format import open_memmap

from sltools.utils import split_seq
from datasets.devisign.recording import Recording


def main():
    assert len(sys.argv) == 2
    datadir = sys.argv[-1]

    print("NOTE: The annotations files should be converted to UTF8 beforehand.")

    print("Reading sequence durations")

    filepaths = []
    durations = []
    for dirpath, _, filenames in os.walk(os.path.join(datadir, "Data", "DEVISIGN_L")):
        for f in filenames:
            print("\r" + f, end='', flush=True)
            filepaths.append(os.path.join(dirpath, f))
            durations.append(Recording(os.path.join(dirpath, f), fast=True).duration)

    durations = np.array(durations)
    file_offsets = np.cumsum(durations) - durations
    subsequences = np.stack([file_offsets, file_offsets + durations], axis=1)

    metadata_dtype = [('signer', 'u1'), ('sess', 'u1'), ('date', 'U8'),
                      ('label', 'i4'), ('duration', 'u4'), ('skel_data_off', 'u8')]
    info = np.empty((len(durations),), dtype=metadata_dtype)

    dump_file = os.path.join(datadir, 'poses_2d.npy')
    storage = open_memmap(dump_file, 'w+', dtype=np.int16,
                          shape=(durations.sum(), 20, 2))
    poses2d = split_seq(storage, subsequences)

    dump_file = os.path.join(datadir, 'poses_3d.npy')
    storage = open_memmap(dump_file, 'w+', dtype=np.float32,
                          shape=(durations.sum(), 20, 3))
    poses3d = split_seq(storage, subsequences)

    for i in range(len(durations)):
        print("\r{} / {}".format(i, len(durations)), end='', flush=True)
        signer, sign, sess, date = Recording.parse_archive_name(filepaths[i])
        r = Recording(os.path.join(datadir, "Data", "DEVISIGN_L"),
                      signer, sign, sess, date)
        info[i]['signer'] = signer
        info[i]['sess'] = sess
        info[i]['date'] = date
        info[i]['label'] = sign
        info[i]['duration'] = r.duration
        info[i]['skel_data_off'] = subsequences[i, 0]
        poses3d[i][...], poses2d[i][...] = r.poses()

    np.save(os.path.join(datadir, 'rec_info.npy'), info)


if __name__ == "__main__":
    main()
