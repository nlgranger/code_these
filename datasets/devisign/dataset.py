import os
import enum
from collections import OrderedDict
import numpy as np
from .recording import Recording


class DEVISIGNDataset:
    class JoinType(enum.IntEnum):
        AnkleLeft = 14,
        AnkleRight = 18,
        ElbowLeft = 5,
        ElbowRight = 9,
        FootLeft = 15,
        FootRight = 19,
        HandLeft = 7,
        HandRight = 11,
        HandTipLeft = 21,
        HandTipRight = 23,
        Head = 3,
        HipLeft = 12,
        HipRight = 16,
        KneeLeft = 13,
        KneeRight = 17,
        Neck = 2,
        ShoulderLeft = 4,
        ShoulderRight = 8,
        SpineBase = 0,
        SpineMid = 1,
        SpineShoulder = 20,
        ThumbLeft = 22,
        ThumbRight = 24,
        WristLeft = 6,
        WristRight = 10
        
    def __init__(self, datadir):
        self._datadir = datadir
        self.rec_info = np.load(os.path.join(self._datadir, 'rec_info.npy'))
        self._poses_2d = np.load(os.path.join(self._datadir, 'poses_2d.npy'),
                                 mmap_mode='r')
        self._poses_3d = np.load(os.path.join(self._datadir, 'poses_3d.npy'),
                                 mmap_mode='r')
        self.cache = OrderedDict()
        self.cache_size = 7

    def __len__(self):
        return len(self.rec_info)

    def _get_rec_wrapper(self, i):
        signer, sess, date, label, _, _ = self.rec_info[i]

        if (signer, sess, date, label) not in self.cache.keys():
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)

            rec = Recording(os.path.join(self._datadir, "Data", "DEVISIGN_L"),
                            signer, sess, date, label)
            self.cache[(signer, sess, date, label)] = rec

        else:
            rec = self.cache[(signer, sess, date, label)]

        return rec

    def positions(self, recording) -> np.ndarray:
        duration = self.rec_info[recording]['duration']
        skel_data_off = self.rec_info[recording]['skel_data_off']

        return np.array(self._poses_2d[skel_data_off:skel_data_off + duration, :, ],
                        np.int32)

    def positions_3d(self, recording) -> np.ndarray:
        duration = self.rec_info[recording]['duration']
        skel_data_off = self.rec_info[recording]['skel_data_off']

        return np.array(self._poses_3d[skel_data_off:skel_data_off + duration, :, ],
                        np.float32)

    def durations(self, recording):
        return self.rec_info[recording]['duration']

    def subject(self, recording):
        raise NotImplementedError

    def labels(self, recording):
        return self.rec_info[recording]['label']

    def bgr_frames(self, recording):
        return self._get_rec_wrapper(recording).bgr_frames()

    def z_frames(self, recording):
        return self._get_rec_wrapper(recording).z_frames()
