import os
from enum import IntEnum
import numpy as np
from lproc import rmap
from ..utils import VideoSequence


class Chalearn2014Dataset:
    class JointType(IntEnum):
        HipCenter = 0,
        Spine = 1,
        ShoulderCenter = 2,
        Head = 3,
        ShoulderLeft = 4,
        ElbowLeft = 5,
        WristLeft = 6,
        HandLeft = 7,
        ShoulderRight = 8,
        ElbowRight = 9,
        WristRight = 10,
        HandRight = 11,
        HipLeft = 12,
        KneeLeft = 13,
        AnkleLeft = 14,
        FootLeft = 15,
        HipRight = 16,
        KneeRight = 17,
        AnkleRight = 18,
        FootRight = 19

    class Coord(IntEnum):
        Wx = 0,
        Wy = 1,
        Wz = 2,
        Rx = 3,
        Ry = 4,
        Rz = 5,
        Rw = 6,
        Px = 7,
        Py = 8

    def __init__(self, datadir):
        self.datadir = datadir
        self.rec_info = np.load(os.path.join(self.datadir, 'rec_info.npy'))
        self.skel_data = np.load(os.path.join(self.datadir, 'skel_data.npy'),
                                 mmap_mode='r')
        self.labels = np.load(os.path.join(self.datadir, 'labels.npy'))

    def __len__(self):
        return len(self.rec_info)

    def positions(self, recording) -> np.ndarray:
        duration = self.rec_info[recording]['duration']
        skel_data_off = self.rec_info[recording]['skel_data_off']

        return self.skel_data[
            skel_data_off:skel_data_off + duration,
            :,
            (self.Coord.Px, self.Coord.Py)].astype(np.int32)

    def positions_3d(self, recording):
        duration = self.rec_info[recording]['duration']
        skel_data_off = self.rec_info[recording]['skel_data_off']

        return self.skel_data[
            skel_data_off:skel_data_off + duration,
            :,
            (self.Coord.Wx, self.Coord.Wy, self.Coord.Wz)].copy()

    def durations(self, recordings):
        return int(self.rec_info[recordings]['duration'].sum())

    def subject(self, recording):
        raise NotImplementedError

    def glosses(self, recording):
        n_labels = self.rec_info[recording]['n_labels']
        lbl_data_off = self.rec_info[recording]['lbl_data_off']

        return self.labels[lbl_data_off:lbl_data_off + n_labels, :].copy()

    def bgr_frames(self, recording):
        num = self.rec_info[recording]['num']
        filename = os.path.join(self.datadir,
                                "Sample{:04d}_color.mp4".format(num))

        return VideoSequence(filename)

    def z_maps(self, recording):
        num = self.rec_info[recording]['num']
        filename = os.path.join(self.datadir,
                                "Sample{:04d}_depth.mp4".format(num))

        max_depth = self.rec_info[recording]['max_depth']

        return rmap(
            lambda frame: frame[:, :, 0].astype(np.float32) * max_depth / 255,
            VideoSequence(filename))

    def default_splits(self):
        return \
            np.where(self.rec_info['split'] == 0)[0], \
            np.where(self.rec_info['split'] == 1)[0], \
            np.where(self.rec_info['split'] == 2)[0]

    def all_recordings(self):
        return np.arange(len(self.rec_info))
