import os
import re
import shutil
import tempfile
import zipfile
import numpy as np
from ..utils import VideoSequence


class Recording:
    def __init__(self, datadir, signer, label, sess, date):
        self.recording_name = "P{:02d}_{:04d}_{:1d}_0_{}.oni".format(
            signer, label, sess, date)

        file_path = os.path.join(
            datadir, "P{:02d}_{}".format(signer, sess),
            self.recording_name + ".zip")
        self.archive = zipfile.ZipFile(file_path)

        with self.archive.open(os.path.join(self.recording_name, 'log.txt'), 'r') as f:
            match = re.fullmatch(
                r"Color: ([0-9]+) frames; [0-9.]+ fps\r\n",
                f.readlines()[1].decode())

            if match is None:
                raise ValueError()

            self.duration = int(match.group(1))

        self._tmpdir = None

    @property
    def tmpdir(self):
        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp()

        return self._tmpdir

    def __del__(self):
        if self._tmpdir is not None:
            shutil.rmtree(self.tmpdir)

        self.archive.close()

    @staticmethod
    def parse_archive_name(name):
        match = re.fullmatch(
            r".*P([0-9]{2})_([0-9]{4})_([0-9])_([0-9])_([0-9]{8}).oni.zip",
            name)
        if match is None:
            raise ValueError("the file name does not correspond to a recording archive")
        if not os.path.isfile(name):
            raise ValueError("the recording archive does not exist or is not a file")

        signer = int(match.group(1))
        label = int(match.group(2))
        sess = int(match.group(3))
        date = match.group(5)

        return signer, label, sess, date

    def bgr_frames(self):
        if not os.path.exists(os.path.join(self.tmpdir, "color.avi")):
            self.archive.extract(self.recording_name + "/color.avi", self.tmpdir)
        return VideoSequence(os.path.join(self.tmpdir, self.recording_name, "color.avi"))

    def z_frames(self):
        if not os.path.exists(os.path.join(self.tmpdir, "depth.dat")):
            self.archive.extract(self.recording_name + "/depth.dat", self.tmpdir)
        return np.fromfile(
            os.path.join(self.tmpdir, self.recording_name, "depth.dat"),
            dtype=np.uint16).reshape((-1, 480, 640))

    def poses(self):
        skeleton_dtype = np.dtype([('3d', np.float32, 4 * 20), ('2d', np.int32, 2 * 20)])

        if not os.path.exists(os.path.join(self.tmpdir, "skeleton.dat")):
            self.archive.extract(self.recording_name + "/skeleton.dat", self.tmpdir)
        data = np.fromfile(
            os.path.join(self.tmpdir, self.recording_name, "skeleton.dat"),
            dtype=skeleton_dtype)

        return data['3d'].reshape((-1, 20, 4))[:, :, :3], data['2d'].reshape((-1, 20, 2))
