#!/usr/bin/env python3

import os
import pickle as pkl
import shelve
from functools import partial

import numpy as np
import lasagne
from lproc import rmap, subset

from datasets.utils import gloss2seq
from sltools.models.rnn import build_predict_fn, build_train_fn
from sltools.nn_utils import compute_scores, seq_hinge_loss  # , seq_ce_loss

from experiments.ch14_skel.a_data import tmpdir, gloss_seqs, durations, \
    train_subset, val_subset
from experiments.ch14_skel.b_preprocess import feat_seqs
from experiments.ch14_skel.c_models import build_lstm

# from experiments.ch14_bgr.a_data import tmpdir, gloss_seqs, durations, \
#     train_subset, val_subset
# from experiments.ch14_bgr.b_preprocess import feat_seqs
# from experiments.ch14_bgr.c_models import build_lstm

# from experiments.ch14_fusion.a_data import tmpdir, gloss_seqs, durations, \
#     train_subset, val_subset
# from experiments.ch14_fusion.b_preprocess import feat_seqs
# from experiments.ch14_fusion.c_models import build_lstm


def main():
    # Data ------------------------------------------------------------------------------

    feat_seqs_train = subset(feat_seqs, train_subset)
    gloss_seqs_train = subset(gloss_seqs, train_subset)
    seqs_durations_train = subset(durations, train_subset)

    feat_seqs_val = subset(feat_seqs, val_subset)
    gloss_seqs_val = subset(gloss_seqs, val_subset)
    seqs_durations_val = subset(durations, val_subset)

    X = feat_seqs_train
    y = rmap(lambda g, d: gloss2seq(g, d, 0), gloss_seqs_train, seqs_durations_train)

    Xv = feat_seqs_val
    yv = rmap(lambda g, d: gloss2seq(g, d, 0), gloss_seqs_val, seqs_durations_val)

    # Model -----------------------------------------------------------------------------

    max_time = 128
    batch_size = 16
    nlabels = 21
    layers = build_lstm(feats_shape=feat_seqs_train[0][0].shape,
                        batch_size=batch_size, max_time=max_time)
    # layers = build_lstm(skel_feats_shape=feat_seqs_train[0][0][0].shape,
    #                     bgr_feats_shape=feat_seqs_train[0][1][0].shape,
    #                     batch_size=batch_size, max_time=max_time)
    warmup = layers.pop('warmup')
    predict_fn = build_predict_fn(layers, batch_size, max_time, nlabels, warmup)

    # Training --------------------------------------------------------------------------

    weights = np.unique(np.concatenate(y), return_counts=True)[1] ** -0.7
    weights *= 21 / weights.sum()
    loss_fn = partial(seq_hinge_loss, weights=weights)
    # loss_fn = partial(seq_ce_loss, weights=weights)
    updates_fn = lasagne.updates.adam
    report = shelve.open(os.path.join(tmpdir, "rnn_report"),
                         protocol=pkl.HIGHEST_PROTOCOL)
    save_every = 5
    resume_at = len(report) - 1 - (len(report) % save_every)
    min_progress = 2e-3  # if improvement is below, decrease learning rate
    l_rate = .001
    train_fn = build_train_fn(
        layers, batch_size, max_time, warmup,
        loss_fn=loss_fn, updates_fn=updates_fn)

    batch_losses = []
    multibatch_losses = np.array([])

    def batch_callback(batch, n_batches, loss):
        batch_losses.append(loss)
        print("\rbatch {:>5d}/{:<5d} loss : {:2.4f}".format(
              batch + 1, n_batches, loss), end='', flush=True)

    for e in range(100):
        # Resume if possible ------------------------------------------------------------

        if e < resume_at:
            print("\repoch {:>5d} : skipped".format(e))
            continue

        if e == resume_at:
            print("\repoch {:>5d} : resumed".format(e))
            epoch_report = report[str(e)]
            min_progress *= epoch_report['l_rate'] / l_rate
            l_rate = epoch_report['l_rate']
            with open(os.path.join(tmpdir, "rnn_it{:04d}.pkl".format(e)), 'rb') as f:
                all_layers = lasagne.layers.get_all_layers(layers['l_linout'])
                params = pkl.load(f)
                lasagne.layers.set_all_param_values(all_layers, params)
            continue

        # Train one epoch ---------------------------------------------------------------

        batch_losses = []
        train_fn(X, y, l_rate=l_rate, n_epochs=1, batch_callback=batch_callback)

        # Generate report ---------------------------------------------------------------

        multibatch_losses = np.concatenate((multibatch_losses, np.asarray(batch_losses)))
        running_loss = batch_losses[0]
        for i in range(1, len(batch_losses)):
            running_loss = .97 * running_loss + .03 * batch_losses[i]
        print("\repoch {:>5d} loss : {:2.4f}       ".format(e + 1, running_loss))

        epoch_report = {
            'batch_losses': batch_losses,
            'epoch_loss': running_loss,
            'l_rate': l_rate
        }

        if (e + 1) % save_every == 0:
            predictions = [np.argmax(p, axis=1) for p in predict_fn(X)]
            j, p, c = compute_scores(predictions, y)
            predictions = [np.argmax(p, axis=1) for p in predict_fn(Xv)]
            jv, pv, cv = compute_scores(predictions, yv)
            epoch_report['train_scores'] = \
                {'jaccard': j, 'framewise': p, 'confusion': c}
            epoch_report['val_scores'] = \
                {'jaccard': jv, 'framewise': pv, 'confusion': cv}

            print("scores: {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}".format(j, p, jv, pv))

            with open(os.path.join(tmpdir, "rnn_it{:04d}.pkl".format(e)), 'wb') as f:
                all_layers = lasagne.layers.get_all_layers(layers['l_linout'])
                params = lasagne.layers.get_all_param_values(all_layers)
                pkl.dump(params, f)

            # Update learning rate ------------------------------------------------------

            b = np.cov(multibatch_losses, np.arange(len(multibatch_losses)))[1, 0] \
                / np.var(np.arange(len(multibatch_losses)))
            print('progress ~= {}'.format(b * len(multibatch_losses)))
            if b * len(multibatch_losses) > - min_progress:
                print("decreasing learning rate: {} -> {}".format(l_rate, l_rate * .3))
                l_rate *= .3
                min_progress *= .3
            multibatch_losses = np.array([])

        report[str(e)] = epoch_report
        report.sync()


if __name__ == "__main__":
    main()
