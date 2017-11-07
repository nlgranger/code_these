import cv2
import numpy as np
from scipy.interpolate import interp1d


class Transformation:
    def __init__(self, ref2d=None, ref3d=None, flip_mapping=None, frame_width=None,
                 fliplr=False, tilt=0, zshift=0, xscale=1, yscale=1, zscale=1, tscale=1):
        self.ref2d = ref2d
        self.ref3d = ref3d
        self.flip_mapping = flip_mapping
        self.frame_width = frame_width
        self.zshift = zshift
        self.fliplr = fliplr
        self.tilt = tilt
        self.xscale = xscale
        self.yscale = yscale
        self.zscale = zscale
        self.tscale = tscale


def transform_pose2d(output, t: Transformation):
    output = np.array(output)
    dtype = output.dtype

    # shorthand notations
    sx, sy = t.xscale, t.yscale
    rx, ry = t.ref2d[:, None, 0], t.ref2d[:, None, 1]
    duration = len(output)

    # z-shift
    if t.zshift is not 0:
        s = 1 / (1 + t.zshift / (t.ref3d[:, None, 2] + .0001))
        output[:, :, 0] = rx * (1 - s) + output[:, :, 0] * s
        output[:, :, 1] = ry * (1 - s) + output[:, :, 1] * s

    # space scale
    output[:, :, 0] = rx * (1 - sx) + output[:, :, 0] * sx
    output[:, :, 1] = ry * (1 - sy) + output[:, :, 1] * sy

    # tilt
    x = output[:, :, 0]
    y = output[:, :, 1]
    x[:, :] = rx + np.cos(t.tilt) * (x - rx) + np.sin(t.tilt) * (y - ry)
    y[:, :] = ry - np.sin(t.tilt) * (x - rx) + np.cos(t.tilt) * (y - ry)

    # fliplr
    if t.fliplr:
        src, dst = t.flip_mapping
        output[:, dst, :] = output[:, src, :]
        output[:, :, 0] = t.frame_width - output[:, :, 0] - 1

    # time scale
    interpolator = interp1d(np.arange(duration), output,
                            axis=0, kind='cubic' if duration > 3 else 'nearest',
                            assume_sorted=True)
    output = interpolator(
        np.linspace(0, duration - 1, transform_durations(duration, t)))\
        .astype(dtype)

    return output


def transform_pose3d(pose3d_seq, t: Transformation):
    output = np.array(pose3d_seq)
    dtype = output.dtype

    # shorthand notations
    sx, sy, sz = t.xscale, t.yscale, t.zscale
    rx, ry, rz = t.ref3d[:, None, 0], t.ref3d[:, None, 1], t.ref3d[:, None, 2]
    x = output[:, :, 0]
    y = output[:, :, 1]
    duration = len(output)

    # z-shift
    output[:, :, 2] += t.zshift

    # space scale
    x[:, :] = rx * (1 - sx) + x * sx
    output[:, :, 1] = ry * (1 - sy) + output[:, :, 1] * sy
    output[:, :, 2] = rz * (1 - sz) + output[:, :, 2] * sz

    # tilt
    x[:, :] = rx + np.cos(t.tilt) * (x - rx) + np.sin(t.tilt) * (y - ry)
    y[:, :] = ry - np.sin(t.tilt) * (x - rx) + np.cos(t.tilt) * (y - ry)

    # fliplr
    if t.fliplr:
        src, dst = t.flip_mapping
        output[:, dst, :] = output[:, src, :]
        output[:, :, 0] = 2 * rx - (output[:, :, 0])

    # time scale
    interpolator = interp1d(np.arange(duration), output,
                            axis=0, kind='cubic' if duration > 3 else 'nearest',
                            assume_sorted=True)
    output = interpolator(
        np.linspace(0, duration - 1, transform_durations(duration, t)))\
        .astype(dtype)

    return output


def transform_frames(frame_seq, t: Transformation):
    frame_seq = np.array(frame_seq)  # heavy on memory!

    # shorthand notations
    duration = len(frame_seq)
    h, w = frame_seq.shape[1:3]
    sx, sy = t.xscale, t.yscale

    # z-shift
    zcorrection = 1 / (1 + t.zshift / (t.ref3d[:, 2] + .0001))

    # space scale, z-shift and tilt
    for f, (rx, ry), zc in zip(frame_seq, t.ref2d, zcorrection):

        pts1 = np.float32([[rx, ry], [rx + 10, ry], [rx, ry + 10]])
        pts2 = np.float32(
            [[rx, ry],
             [rx + sx  * zc * 10 * np.cos(t.tilt), ry - sx * zc * 10 * np.sin(t.tilt)],
             [rx + sy * zc * 10 * np.sin(t.tilt), ry + sy * zc * 10 * np.cos(t.tilt)]])
        tmatrix = cv2.getAffineTransform(pts1, pts2)
        f[...] = cv2.warpAffine(f, tmatrix, (w, h))

    # fliplr
    if t.fliplr:
        frame_seq = np.flip(frame_seq, axis=1)

    # time scale
    if t.tscale != 1:
        indexes = np.round(
            np.linspace(0, duration - 1, transform_durations(duration, t)))
        indexes = indexes.astype(np.int)
        frame_seq = frame_seq[indexes]

    return frame_seq


def transform_glosses(gloss_seq, duration, t: Transformation):
    gloss_seq = np.copy(gloss_seq)
    gloss_seq[:, 1:3] = np.fmin(gloss_seq[:, 1:3] / t.tscale,
                                transform_durations(duration, t))
    return gloss_seq


def transform_durations(duration, t: Transformation):
    return int(duration / t.tscale)
