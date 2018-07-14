from typing import Sequence
import av


def split_seq(seq, glosses, offset=0, min_len=1, words=None):
    """Split a sequence by glosses.

    :param seq:
        sequence of labels. If seq is a multidimensional array, splitting
        occurs along the first dimension.
    :param glosses:
        list of glosses for seq
    :param words:
        accepted gloss labels (other glosses are ignored)
    :param offset:
        time of the first element of seq
    :param min_len:
        if possible, take samples before and after the gloss to reach min_len elements
    :return (list, list):
        A list of subsequences and a list of the corresponding gloss values.
    """
    subseqs = []
    labels = []
    for g in glosses:
        if words is None or g.value in words:
            start = g.start - offset
            stop = g.stop - offset
            if start >= len(seq) or stop < 0:  # seq too short or offset too large
                continue

            duration = stop - start
            if duration <= min_len:
                start -= round((min_len - duration)/2)
                stop += min_len - duration - round((min_len - duration)/2)
            start = max(0, start)
            stop = min(len(seq), stop)

            subseqs.append(seq[start:stop, ...])
            labels.append(g.value)
    return subseqs, labels


def split_dataset_seqs(dataset, sequences, *kargs, **kwargs):
    """Split a set of sequences into subsequences corresponding to glosses.

    :param dataset:
        The dataset containing glossing annotations.
    :param sequences:
        A list of (recording id, sequence) tuples
    Extraneaous parameters are forwared to :ref:`split_seq`
    :return:
        A list of subsequences and a list of the corresponding gloss values.
    """
    x = []
    y = []
    for rec, seq in sequences:
        subseqs, labels = split_seq(seq, dataset.glosses(rec),
                                    *kargs, **kwargs)
        x.extend(subseqs)
        y.extend(labels)

    return x, y


# def gloss2seq(glosses, duration, default=-1, dtype=np.int32):
#     seq = np.full((duration,), default, dtype=dtype)
#     for l, start, stop in glosses:
#         if start < 0 or stop > duration:
#             raise ValueError
#         seq[start:stop] = l
#     return seq


# def seq2gloss(seq):
#     glosses = []
#
#     current = seq[0]
#     start = 0
#     for i, v in enumerate(seq):
#         if v != current:
#             glosses.append((current, start, i))
#             current = v
#             start = i
#
#     if start < len(seq):
#         glosses.append((current, start, len(seq)))
#
#     return glosses


# class VideoSequence(Sequence):
#     def __init__(self, filename):
#         self.filename = filename
#         self.video = cv2.VideoCapture(filename)
#         self.start = 0
#         self.stop = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
#
#         if self.stop == 0:
#             raise ValueError("{} is empty!".format(filename))
#
#     def __len__(self):
#         return self.stop - self.start
#
#     def __iter__(self):
#         for t in range(len(self)):
#             yield self[t]
#
#     def __getstate__(self):
#         return self.filename, self.start, self.stop
#
#     def __setstate__(self, state):
#         filename, start, stop = state
#         self.__init__(filename)
#         self.start = start
#         self.stop = stop
#
#     def __getitem__(self, item):
#         if isinstance(item, slice):
#             start, stop, step = item.start, item.stop, item.step
#
#             if step is not None and step != 1:
#                 raise ValueError("Only step = 1 is supported")
#
#             if start is None:
#                 start = self.start
#             elif start < 0:
#                 start += self.stop
#             else:
#                 start += self.start
#             if stop is None:
#                 stop = self.stop
#             elif stop < 0:
#                 stop += self.stop
#             else:
#                 stop += self.start
#
#             sliced = VideoSequence(self.filename)
#             sliced.start = start
#             sliced.stop = stop
#             return sliced
#
#         else:
#             if item < -len(self) or item >= len(self):
#                 raise IndexError("Video frame {} is out of range.".format(item))
#
#             t = self.start + item if item >= 0 else self.stop + item
#
#             if self.video.get(cv2.CAP_PROP_POS_FRAMES) != t:
#                 self.video.set(cv2.CAP_PROP_POS_FRAMES, t)
#
#             ok, frame = self.video.read()
#             if not ok:
#                 raise ValueError("Failed to read frame.")
#
#             return frame


class VideoSequence(Sequence):
    def __init__(self, file):
        self.file = file
        self.container = av.open(file)
        self.stream = self.container.streams.get(video=0)[0]
        self.frame_base = self.stream.time_base * self.stream.average_rate
        self.packet_iter = self.container.demux(self.stream)
        self.last_packet = next(self.packet_iter).decode()

        self.offset = 0
        self.duration = int(self.stream.duration * self.stream.time_base
                            * self.stream.average_rate)

    def __len__(self):
        return self.duration

    def __getitem__(self, t):
        # slicing support
        if isinstance(t, slice):
            start, stop, step = t.start, t.stop, t.step

            # defaults
            start = start or 0
            stop = stop or -1
            step = step or 1

            # range check
            if step != 1:
                raise IndexError("VideoSequence slicing is limited to step 1")
            if start < -len(self) or start >= len(self) \
                    or stop < -len(self) - 1 or stop > len(self):
                raise IndexError("VideoSequence slice index out of range.")

            # negative indexing
            start = start + len(self) if start < 0 else start
            stop = stop + len(self) if stop < 0 else stop
            stop = max(start, stop)

            video = VideoSequence(self.file)
            video.offset = self.offset + start  # cumulate offsets
            video.duration = stop - start

            return video

        # range check
        if t < -len(self) or t >= len(self):
            raise IndexError("VideoSequence index out of range.")

        # negative indexing
        if t < 0:
            t += len(self)

        t += self.offset

        t_pts = t / self.stream.time_base / self.stream.average_rate

        # Do seeking if needed
        if t > self.last_packet[-1].pts * self.frame_base + len(self.last_packet) \
                or t < self.last_packet[0].pts * self.frame_base:
            self.stream.seek(int(t / self.frame_base))
            self.packet_iter = self.container.demux(self.stream)

        while True:
            for f in self.last_packet:
                if f.pts == t_pts:
                    return f.to_rgb().to_nd_array()
            self.last_packet = next(self.packet_iter).decode()
