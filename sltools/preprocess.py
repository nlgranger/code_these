import cv2
import numpy as np
from scipy.interpolate import interp1d


def interpolate_positions(p, invalid, kind='cubic'):
    valid = np.invert(invalid)

    for j in range(p.shape[1]):
        x = np.where(invalid[:, j])[0]
        xp = np.where(valid[:, j])[0]

        if len(xp) < 2:
            continue

        p[x, j, :] = interp1d(
            xp, p[xp, j, :], kind=kind,
            bounds_error=False, fill_value=(p[xp[0], j], p[xp[-1], j]),
            axis=0, assume_sorted=True)(x)

    return p


def fix_contrast(video):
    video = np.asarray(video)
    clahe = cv2.createCLAHE(clipLimit=.50, tileGridSize=(32, 32))
    d, h, w, _ = video.shape
    output = np.empty((d, h, w), dtype=np.float32)

    for t in range(d):
        output[t] = clahe.apply(video[t, :, :, 0])
        output[t] += clahe.apply(video[t, :, :, 1])
        output[t] += clahe.apply(video[t, :, :, 2])

    output /= 3 * 255
    return output