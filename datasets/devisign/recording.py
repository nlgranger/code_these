import os
import re
import shutil
import tempfile
import zipfile
import numpy as np
from ..utils import VideoSequence


class Recording:
    def __init__(self, datadir, signer, sess, date, label, fast=False):
        self.recording_name = "P{:02d}_{:04d}_{}_0_{}.oni".format(
            signer, label, sess, date)
        file_path = os.path.join(
            datadir, "P{:02d}_{}".format(signer, sess),
            self.recording_name + ".zip")

        with zipfile.ZipFile(file_path) as z:
            with z.open(os.path.join(self.recording_name, 'log.txt'), 'r') as f:
                match = re.fullmatch(
                    r"Color: ([0-9]+) frames; [0-9\.]+ fps\r\n",
                    f.readlines()[1].decode())
                assert match is not None
                self.duration = int(match.group(1))

            if not fast:
                self.tmpdir = tempfile.mkdtemp()
                z.extractall(self.tmpdir)
            else:
                self.tmpdir = None

    def __del__(self):
        if self.tmpdir is not None:
            shutil.rmtree(self.tmpdir)

    @staticmethod
    def parse_archive_name(name):
        match = re.fullmatch(
            r".*P([0-9]{2,2})_([0-9]{4,4})_([0-9])_([0-9])_([0-9]{8,8}).oni.zip",
            name)
        if match is None:
            raise ValueError("the file name does not correspond to a recording archive")
        if not os.path.isfile(name):
            raise ValueError("the recording archive does not exist or is not a file")

        signer = int(match.group(1))
        label = int(match.group(2))
        sess = int(match.group(3))
        date = match.group(5)

        return signer, sess, date, label

    def bgr_frames(self):
        return VideoSequence(os.path.join(self.tmpdir, self.recording_name, 'color.avi'))

    def z_frames(self):
        return np.fromfile(
            os.path.join(self.tmpdir, self.recording_name, 'depth.dat'),
            dtype=np.uint16).reshape((-1, 480, 640))

    def poses(self):
        skeleton_dtype = np.dtype([('3d', np.float32, 4 * 20), ('2d', np.int32, 2 * 20)])

        data = np.fromfile(
            os.path.join(self.tmpdir, self.recording_name, 'skeleton.dat'),
            dtype=skeleton_dtype)

        return data['3d'].reshape((-1, 20, 4))[:, :, :3], data['2d'].reshape((-1, 20, 2))
