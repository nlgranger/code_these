#!/bin/env python3

import lasagne
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
from lproc import rmap, subset

from experiments.siamese_oneshot.a_data import durations, labels, \
    train_subset, val_subset
from experiments.siamese_oneshot.b_preprocess import feat_seqs
from experiments.siamese_oneshot.c_model import build_model
from sltools.nn_utils import adjust_length

max_time = 128
batch_size = 32
l_rate = 0.0001


class MinibatchGenerator:
    def __init__(self, data, labels, batch_size, positive_ratio=.5):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio

        sample = data[0]
        shapes = [(batch_size,) + np.asarray(d).shape for d in sample]
        dtypes = [np.asarray(d).dtype for d in sample]

        self.buffers_l = [np.empty(s, dtype=d) for s, d in zip(shapes, dtypes)]
        self.buffers_r = [np.empty(s, dtype=d) for s, d in zip(shapes, dtypes)]

        self.indexes = [[slice(None)] * len(s) for s in shapes]
        self.labels = labels
        self.where_labels = {l: (np.where(labels == l)[0], np.where(labels != l)[0])
                             for l in labels}

    def __iter__(self):
        left_shuffle = np.random.permutation(len(self.labels))
        for i in range(0, len(left_shuffle) - len(left_shuffle) % self.batch_size,
                       self.batch_size):
            yield self._gen_minibatch(left_shuffle[i:i + self.batch_size])

    def _gen_minibatch(self, left_items):
        for b in self.buffers_l:
            b.fill(0)
        for b in self.buffers_r:
            b.fill(0)

        for b in range(len(left_items)):
            data = self.data[left_items[b]]
            for ax in range(len(self.buffers_l)):
                self.buffers_l[ax][b] = data[ax]

        left_labels = self.labels[left_items]

        right_items = np.zeros((self.batch_size,), dtype=np.int)
        positive = np.zeros((self.batch_size,), dtype=np.float32)

        for b in range(len(left_items)):
            positive[b] = np.random.random() < self.positive_ratio
            where = self.where_labels[left_labels[b]][0 if positive[b] else 1]
            right_items[b] = where[np.random.randint(0, len(where))]

        for b in range(len(left_items)):
            data = self.data[right_items[b]]
            for ax in range(len(self.buffers_l)):
                self.buffers_r[ax][b] = data[ax]

        return self.buffers_l, self.buffers_r, positive


def main():
    # Load dataset
    combined_data = rmap(lambda x, d: (adjust_length(x, max_time), min(d, max_time)),
                         feat_seqs, durations)
    X = subset(combined_data, train_subset)
    y = labels[train_subset]
    Xv = subset(combined_data, val_subset)
    yv = labels[val_subset]
    train_batches = MinibatchGenerator(X, y, batch_size)
    val_batches = MinibatchGenerator(Xv, yv, batch_size)

    # Build model
    model = build_model(feat_seqs[0][0].shape, batch_size, max_time)
    l_linout = model['l_linout']
    l_in_left, l_in_right = model['l_in']
    l_duration_left, l_duration_right = model['l_duration']
    linout = lasagne.layers.get_output(l_linout)

    # train_routine
    targets = T.vector('targets')
    l_rate_var = T.scalar('l_rate')
    loss = T.switch(targets > .1,
                    .5 * linout ** 2,
                    .5 * T.maximum(0, 1 - linout) ** 2).sum()
    params = lasagne.layers.get_all_params(l_linout, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=l_rate_var)
    update_fn = theano.function([l_in_left.input_var, l_duration_left.input_var,
                                 l_in_right.input_var, l_duration_right.input_var,
                                 targets, l_rate_var],
                                outputs=loss, updates=updates)

    linout2 = lasagne.layers.get_output(l_linout, deterministic=True)
    loss2 = T.switch(targets > .1,
                     .5 * linout2 ** 2,
                     .5 * T.maximum(0, 1 - linout2) ** 2)
    predict_fn = theano.function([l_in_left.input_var, l_duration_left.input_var,
                                  l_in_right.input_var, l_duration_right.input_var,
                                  targets],
                                 outputs=[linout2, loss2])

    running_loss = 0
    for e in range(900):
        if e % 5 == 0:
            all_preds = np.empty((len(y,) - len(y) % batch_size))
            all_losses = np.empty((len(y,) - len(y) % batch_size))
            all_targets = np.empty((len(y,) - len(y) % batch_size))
            i = 0
            for (xl, dl), (xr, dr), tgt in train_batches:
                all_preds[i:i + len(tgt)], all_losses[i:i + len(tgt)] = \
                    predict_fn(xl, dl, xr, dr, tgt)
                all_targets[i:i + len(tgt)] = tgt
                i += len(tgt)

            plt.figure()
            bins = np.linspace(0, max(all_preds), 40)
            pos, _ = np.histogram(all_preds[all_targets == 1], bins=bins)
            neg, _ = np.histogram(all_preds[all_targets == 0], bins=bins)
            plt.bar(bins[:-1], pos, width=.025, color='red', alpha=.5)
            plt.bar(bins[:-1], neg, width=.025, color='blue', alpha=.5)
            plt.show()
            plt.figure()
            plt.scatter(all_preds[all_targets > .1], all_losses[all_targets > .1],
                        color='red')
            plt.scatter(all_preds[all_targets < .1], all_losses[all_targets < .1],
                        color='blue')
            plt.show()

            all_preds = np.empty((len(yv,) - len(yv) % batch_size))
            all_losses = np.empty((len(yv,) - len(yv) % batch_size))
            all_targets = np.empty((len(yv,) - len(yv) % batch_size))
            i = 0
            for (xl, dl), (xr, dr), tgt in val_batches:
                all_preds[i:i + len(tgt)], all_losses[i:i + len(tgt)] = \
                    predict_fn(xl, dl, xr, dr, tgt)
                all_targets[i:i + len(tgt)] = tgt
                i += len(tgt)

            plt.figure()
            bins = np.linspace(0, max(all_preds), 40)
            pos, _ = np.histogram(all_preds[all_targets == 1], bins=bins)
            neg, _ = np.histogram(all_preds[all_targets == 0], bins=bins)
            plt.bar(bins[:-1], pos, width=.025, color='red', alpha=.5)
            plt.bar(bins[:-1], neg, width=.025, color='blue', alpha=.5)
            plt.show()
            plt.figure()
            plt.scatter(all_preds[all_targets > .1], all_losses[all_targets > .1],
                        color='red')
            plt.scatter(all_preds[all_targets < .1], all_losses[all_targets < .1],
                        color='blue')
            plt.show()

        for (xl, dl), (xr, dr), tgt in train_batches:
            batch_loss = update_fn(xl, dl, xr, dr, tgt, l_rate)
            running_loss = .92 * running_loss + .02 * batch_loss
            print("\rloss: {}".format(batch_loss), end="", flush=True)

        print("\repoch {:3d} loss: {}".format(e, running_loss))


if __name__ == "__main__":
    main()
