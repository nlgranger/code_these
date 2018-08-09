import cv2
import numpy as np
from scipy.interpolate import interp1d
import seqtools


TransformationType = np.dtype([
    ('ref2d', (np.float32, 2)),
    ('ref3d', (np.float32, 3)),
    ('fliplr', np.bool),
    ('tilt', np.float32),
    ('zshift', np.float32),
    ('xscale', np.float32),
    ('yscale', np.float32),
    ('zscale', np.float32),
    ('tscale', np.float32)])


class Transformation:
    def __init__(self, ref2d=None, ref3d=None, frame_width=None,
                 fliplr=False, tilt=0, zshift=0, xscale=1, yscale=1, zscale=1, tscale=1):
        self.ref2d = ref2d
        self.ref3d = ref3d
        self.frame_width = frame_width
        self.zshift = zshift
        self.fliplr = fliplr
        self.tilt = tilt
        self.xscale = xscale
        self.yscale = yscale
        self.zscale = zscale
        self.tscale = tscale


def transform_pose2d(pose2d_seq, t: Transformation, flip_mapping, frame_width):
    output = np.copy(pose2d_seq).astype(np.float)

    # shorthand notations
    sx, sy = t.xscale, t.yscale
    if len(t.ref2d.shape) == 2:
        rx, ry, rz = t.ref2d[:, None, 0], t.ref2d[:, None, 1], t.ref3d[:, None, 2]
    else:
        rx, ry, rz = \
            t.ref2d[None, None, 0], t.ref2d[None, None, 1], t.ref3d[None, None, 2]
    duration = len(output)
    x = output[:, :, 0]
    y = output[:, :, 1]

    # z-shift
    z_corrections = 1 + t.zshift / (rz + .0001)
    x[...] = (x - rx) / z_corrections + rx
    y[...] = (y - ry) / z_corrections + ry

    # space scale
    x[...] = (x - rx) * sx + rx
    y[...] = (y - ry) * sy + ry

    # tilt
    x[...] = rx + np.cos(t.tilt) * (x - rx) - np.sin(t.tilt) * (y - ry)
    y[...] = ry + np.sin(t.tilt) * (x - rx) + np.cos(t.tilt) * (y - ry)

    # fliplr
    if t.fliplr:
        src, dst = flip_mapping
        output[:, dst, :] = output[:, src, :]
        x[...] = frame_width - x - 1

    # time scale
    interpolator = interp1d(np.arange(duration), output,
                            axis=0, kind='cubic' if duration > 3 else 'nearest',
                            assume_sorted=True)
    output_duration = transform_durations(duration, t)
    output = interpolator(np.linspace(0, duration - 1, output_duration))

    return output.astype(pose2d_seq[0].dtype)


def transform_pose3d(pose3d_seq, t: Transformation, flip_mapping):
    output = np.copy(pose3d_seq).astype(np.float)

    # shorthand notations
    sx, sy, sz = t.xscale, t.yscale, t.zscale
    if len(t.ref2d.shape) == 2:
        rx, ry, rz = t.ref2d[:, None, 0], t.ref2d[:, None, 1], t.ref3d[:, None, 2]
    else:
        rx, ry, rz = \
            t.ref2d[None, None, 0], t.ref2d[None, None, 1], t.ref3d[None, None, 2]
    x = output[:, :, 0]
    y = output[:, :, 1]
    z = output[:, :, 2]
    duration = len(output)

    # z-shift
    z[...] += t.zshift

    # space scale
    x[...] = rx * (1 - sx) + x * sx
    y[...] = ry * (1 - sy) + y * sy
    z[...] = rz * (1 - sz) + z * sz

    # tilt
    x[...] = rx + np.cos(t.tilt) * (x - rx) - np.sin(t.tilt) * (y - ry)
    y[...] = ry + np.sin(t.tilt) * (x - rx) + np.cos(t.tilt) * (y - ry)

    # fliplr
    if t.fliplr:
        src, dst = flip_mapping
        output[:, dst, :] = output[:, src, :]
        x[...] = 2 * rx - x

    # time scale
    interpolator = interp1d(np.arange(duration), output,
                            axis=0, kind='cubic' if duration > 3 else 'nearest',
                            assume_sorted=True)
    output_duration = transform_durations(duration, t)
    output = interpolator(np.linspace(0, duration - 1, output_duration))

    return output.astype(pose3d_seq[0].dtype)


def transform_frames(frame_seq, t: Transformation):
    # shorthand notations
    duration = len(frame_seq)
    sx, sy = t.xscale, t.yscale
    rx, ry = t.ref2d
    rz = t.ref3d[2]

    # generate affine transformation matrix
    # triangles_src = np.array([[rx, rx + 1, rx], [ry, ry, ry + 1]], dtype=np.float32)
    triangles_src = np.array([[rx, ry], [rx + 10, ry], [rx, ry + 10]], dtype=np.float32)
    triangles_dst = np.copy(triangles_src)
    x = triangles_dst[:, 0]
    y = triangles_dst[:, 1]

    z_corrections = 1 + t.zshift / (rz + .0001)
    x[...] = (x - rx) / z_corrections + rx
    y[...] = (y - ry) / z_corrections + ry

    x[...] = (x - rx) * sx + rx
    y[...] = (y - ry) * sy + ry

    x[...] = rx + np.cos(t.tilt) * (x - rx) - np.sin(t.tilt) * (y - ry)
    y[...] = ry + np.sin(t.tilt) * (x - rx) + np.cos(t.tilt) * (y - ry)

    tmatrix = cv2.getAffineTransform(triangles_src, triangles_dst)

    # affine frame-wise transformations
    output = seqtools.smap(lambda f: cv2.warpAffine(f, tmatrix, (640, 480)), frame_seq)

    # fliplr
    if t.fliplr:
        output = seqtools.smap(np.fliplr, output)

    # time scale
    if t.tscale != 1:
        output_duration = transform_durations(duration, t)
        indices = np.round(np.linspace(0, duration - 1, output_duration)).astype(np.int)
        output = seqtools.gather(output, indices)
        if t.tscale > 1:
            output = seqtools.add_cache(output, cache_size=1)

    return output


def transform_glosses(gloss_seq, duration, t: Transformation):
    gloss_seq = np.copy(gloss_seq)
    gloss_seq[:, 1:3] = np.fmin(gloss_seq[:, 1:3] / t.tscale,
                                transform_durations(duration, t))
    return gloss_seq


def transform_durations(duration, t: Transformation):
    return int(duration / t.tscale)
