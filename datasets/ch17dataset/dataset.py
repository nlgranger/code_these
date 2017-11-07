import os
from enum import IntEnum
import numpy as np
from .utils import VideoSequence


class Chalearn2017Dataset:
    class JointType(IntEnum):
        Nose = 0,
        Neck = 1,
        ShoulderRight = 2,
        ElbowRight = 3,
        WristRight = 4,
        ShoulderLeft = 5,
        ElbowLeft = 6,
        WristLeft = 7,
        HipRight = 8,
        KneeRight = 9,
        AnkleRight = 10,
        HipLeft = 11,
        KneeLeft = 12,
        AnkleLeft = 13,
        EyeRight = 14,
        EyeLeft = 15,
        EarRight = 16,
        EarLeft = 17,
        Bkg = 18

    class Coord(IntEnum):
        Px = 0,
        Py = 1

    def __init__(self, datadir):
        self.datadir_ = datadir
        self.rec_info_ = np.load(os.path.join(self.datadir_, 'rec_info.npy'))
        self.poses_ = np.load(os.path.join(self.datadir_, 'poses.npy'),
                              mmap_mode='r')
        self.glosses_ = np.load(os.path.join(self.datadir_, 'glosses.npy'))

    def __len__(self):
        return len(self.rec_info_)

    def positions(self, recording) -> np.ndarray:
        duration = self.rec_info_[recording]['duration']
        skel_data_off = self.rec_info_[recording]['skel_data_off']

        return self.poses_[
            skel_data_off:skel_data_off + duration,
            :,
            (self.Coord.Px, self.Coord.Py)].astype(np.int)

    def durations(self, recordings):
        return int(self.rec_info_[recordings]['duration'].sum())

    def subject(self, recording):
        raise NotImplementedError

    def glosses(self, recording):
        n_labels = self.rec_info_[recording]['n_labels']
        lbl_data_off = self.rec_info_[recording]['lbl_data_off']

        return self.glosses_[lbl_data_off:lbl_data_off + n_labels, :].copy()

    def bgr_frames(self, recording):
        recpath = self.rec_info_[recording]['id']
        filename = os.path.join(self.datadir_, "videos", "{}.M.avi".format(recpath))

        return VideoSequence(filename)

    def default_splits(self):
        return \
            np.where(np.chararray.startswith(self.rec_info_['id'], 'train'))[0], \
            np.where(np.chararray.startswith(self.rec_info_['id'], 'valid'))[0], \
            np.where(np.chararray.startswith(self.rec_info_['id'], 'test'))[0]
