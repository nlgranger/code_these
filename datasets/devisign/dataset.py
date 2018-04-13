import os
import enum
import numpy as np
from .recording import Recording


class DEVISIGNDataset:
    class JointType(enum.IntEnum):
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

    def __init__(self, datadir, cachedir):
        self._datadir = datadir
        self._cachedir = cachedir
        self.rec_info = np.load(os.path.join(self._cachedir, 'rec_info.npy'))
        self._poses_2d = np.load(os.path.join(self._cachedir, 'poses_2d.npy'),
                                 mmap_mode='r')
        self._poses_3d = np.load(os.path.join(self._cachedir, 'poses_3d.npy'),
                                 mmap_mode='r')

    def __len__(self):
        return len(self.rec_info)

    def _get_rec_wrapper(self, i):
        signer, sess, date, label, _, _ = self.rec_info[i]

        return Recording(self._datadir,
                         signer, label, sess, date)

    def positions(self, recording) -> np.ndarray:
        duration = self.rec_info['duration'][recording]
        skel_data_off = self.rec_info['skel_data_off'][recording]

        return np.array(self._poses_2d[skel_data_off:skel_data_off + duration, :, ],
                        np.int32)

    def positions_3d(self, recording) -> np.ndarray:
        duration = self.rec_info['duration'][recording]
        skel_data_off = self.rec_info['skel_data_off'][recording]

        return np.array(self._poses_3d[skel_data_off:skel_data_off + duration, :, ],
                        np.float32)

    def durations(self, recording):
        return self.rec_info['duration'][recording]

    def subject(self, recording):
        return self.rec_info['signer'][recording]

    def label(self, recording):
        return self.rec_info['label'][recording]

    def bgr_frames(self, recording):
        return self._get_rec_wrapper(recording).bgr_frames()

    def z_frames(self, recording):
        return self._get_rec_wrapper(recording).z_frames()

    def split_500(self):
        del self

        return np.array([
            0,    2,    6,    7,    9,    10,   12,   13,   23,   38,   48,
            49,   52,   53,   55,   56,   61,   62,   67,   70,   73,   74,
            82,   88,   98,   99,   111,  117,  126,  129,  130,  135,  137,
            139,  144,  146,  149,  153,  159,  170,  171,  172,  184,  190,
            197,  203,  241,  268,  292,  316,  319,  326,  328,  348,  349,
            351,  352,  356,  359,  360,  379,  388,  393,  395,  449,  522,
            524,  526,  539,  547,  555,  558,  572,  577,  579,  586,  600,
            617,  632,  640,  645,  659,  660,  661,  664,  665,  667,  668,
            670,  672,  684,  687,  688,  689,  693,  694,  699,  700,  701,
            702,  713,  717,  745,  758,  760,  776,  784,  790,  792,  793,
            805,  821,  823,  827,  828,  829,  830,  832,  837,  839,  856,
            860,  871,  890,  918,  933,  938,  946,  953,  968,  982,  995,
            1009, 1013, 1014, 1019, 1020, 1022, 1023, 1024, 1025, 1026, 1027,
            1029, 1030, 1080, 1136, 1149, 1171, 1176, 1179, 1203, 1204, 1205,
            1207, 1208, 1213, 1237, 1239, 1241, 1244, 1253, 1258, 1260, 1263,
            1269, 1275, 1277, 1294, 1297, 1299, 1300, 1301, 1304, 1313, 1314,
            1319, 1327, 1334, 1337, 1343, 1346, 1349, 1351, 1352, 1353, 1360,
            1361, 1372, 1379, 1381, 1382, 1386, 1387, 1388, 1389, 1391, 1393,
            1395, 1398, 1399, 1400, 1404, 1427, 1452, 1467, 1471, 1473, 1474,
            1478, 1479, 1480, 1484, 1490, 1493, 1504, 1505, 1514, 1515, 1521,
            1522, 1524, 1525, 1533, 1535, 1547, 1556, 1568, 1569, 1572, 1583,
            1588, 1603, 1633, 1672, 1695, 1702, 1722, 1723, 1729, 1731, 1732,
            1746, 1763, 1764, 1765, 1773, 1802, 1820, 1847, 1854, 1904, 1971,
            2036, 2088, 2093, 2094, 2140, 2146, 2148, 2164, 2170, 2183, 2184,
            2186, 2188, 2198, 2200, 2208, 2217, 2220, 2328, 2334, 2337, 2348,
            2352, 2365, 2392, 2398, 2416, 2468, 2476, 2479, 2503, 2506, 2550,
            2588, 2612, 2616, 2692, 2702, 2717, 2718, 2731, 2768, 2769, 2775,
            2779, 2789, 2816, 2825, 2843, 2857, 2858, 2862, 2868, 2869, 2871,
            2872, 2874, 2882, 2888, 2891, 2893, 2894, 2895, 2896, 2897, 2900,
            2904, 2914, 2924, 2954, 2970, 2981, 2984, 2985, 2993, 3005, 3011,
            3037, 3122, 3136, 3169, 3181, 3184, 3204, 3206, 3207, 3212, 3216,
            3218, 3220, 3247, 3269, 3290, 3292, 3296, 3306, 3318, 3320, 3370,
            3372, 3387, 3432, 3438, 3471, 3486, 3504, 3505, 3506, 3508, 3509,
            3522, 3531, 3535, 3536, 3538, 3542, 3548, 3553, 3554, 3556, 3558,
            3559, 3563, 3565, 3566, 3575, 3576, 3612, 3627, 3637, 3672, 3678,
            3696, 3747, 3754, 3766, 3767, 3777, 3789, 3800, 3801, 3814, 3817,
            3847, 3859, 3860, 3865, 3883, 3888, 3895, 3903, 3946, 3956, 3972,
            3976, 4007, 4012, 4015, 4033, 4064, 4065, 4068, 4086, 4141, 4146,
            4147, 4151, 4166, 4169, 4188, 4205, 4222, 4245, 4293, 4299, 4303,
            4305, 4309, 4317, 4318, 4319, 4320, 4323, 4324, 4325, 4327, 4328,
            4329, 4332, 4336, 4337, 4338, 4339, 4341, 4342, 4343, 4344, 4345,
            4347, 4348, 4350, 4360, 4361, 4364, 4365, 4369, 4371, 4373, 4375,
            4376, 4377, 4378, 4379, 4380, 4381, 4382, 4383, 4384, 4385, 4386,
            4387, 4388, 4389, 4390, 4391, 4392, 4393, 4394, 4395, 4396, 4397,
            4398, 4399, 4400, 4401, 4402, 4403, 4404, 4405, 4406, 4407, 4408,
            4409, 4410, 4411, 4412, 4413])
        # subset = np.isin(self.rec_info["label"], label_subset)
        # return np.where(subset)[0]
