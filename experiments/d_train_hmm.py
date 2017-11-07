#!/usr/bin/env python3

import os
import pickle as pkl
import shelve
from collections import Counter

import numpy as np
from lproc import rmap, subset, SerializableFunc

from ch14dataset import gloss2seq
from sltools.models import HMMRecognizer, PosteriorModel
from sltools.nn_utils import compute_scores


# Helper --------------------------------------------------------------------------------

def epoch_perfs(model, feats_seqs, gloss_seqs, seqs_durations, previous_model):
    report = {}

    # Complete model
    preds = model.predict(feats_seqs)
    labels = [gloss2seq(g_, d_, 0) for g_, d_ in zip(gloss_seqs, seqs_durations)]
    report['jaccard'], report['framewise'], report['confusion'] = \
        compute_scores(preds, labels)

    # State-wise
    preds = [np.argmax(model.posterior.predict_proba(x), axis=1)
             for x in feats_seqs]
    if previous_model is None:  # fresh start -> hard label assignment
        seqs_duration = rmap(len, feats_seqs)
        state_labels = rmap(model._linearstateassignment, gloss_seqs, seqs_duration)
    else:
        states = rmap(lambda f, g: previous_model._supervized_state_alignment(f, g),
                      feats_seqs, gloss_seqs)
        state_labels = rmap(lambda s: previous_model.state2idx[s], states)

    report['statewise_jaccard'], report['statewise_framewise'], \
        report['statewise_confusion'] = compute_scores(preds, state_labels)

    # Posterior model
    idx2labels = np.concatenate(
        [np.full((model.chains_lengths[i],), model.labels[i])
         for i in range(model.nlabels)] + [np.zeros((1,))]).astype(np.int32)
    preds = [idx2labels[p] for p in preds]

    report['posterior_jaccard'], report['posterior_framewise'], \
        report['posterior_confusion'] = compute_scores(preds, labels)

    return report


batch_losses = []
epoch_losses = []


@SerializableFunc
def callback(epoch, n_epochs, batch, n_batches, loss):
    batch_losses.append(loss)
    print("\rbatch {:>5d}/{:<5d} loss : {:2.4f}".format(
        batch + 1, n_batches, loss), end='', flush=True)

    if batch + 1 == n_batches:
        epoch_loss = np.mean(batch_losses[-n_batches // 10:])
        epoch_losses.append(epoch_loss)
        print("\repoch {:>5d}/{:<5d} loss : {:2.4f}".format(
            epoch + 1, n_epochs, epoch_loss))


# Parametric settings -------------------------------------------------------------------

@SerializableFunc
def compute_weights(X, y, chunks, n_states):
    del X
    counts = Counter({i: 0 for i in range(n_states)})
    for seq, start, stop in chunks:
        counts.update(y[seq][start:stop])
    counts = np.asarray(sorted(counts.items()))[:, 1]
    weights = np.power((counts + 100) / counts.max(), -0.7)
    weights = weights / np.sum(weights) * n_states
    # print(np.array2string(counts, precision=2))
    # print(np.array2string(weights, precision=2))
    return weights


# @SerializableFunc
# def filter_chunks(X, y, chunks, idle_state):
#     del X
#     return [(seq, start, stop) for seq, start, stop
#             in chunks  # but discard mostly idle segments
#             if np.mean(np.equal(y[seq][start:stop], idle_state)) < .6]

@SerializableFunc
def filter_chunks(X, y, chunks, idle_state):
    del X, y, idle_state
    return chunks


# Training script -----------------------------------------------------------------------

def main():
    from experiments.ch14_skel.a_data import tmpdir, gloss_seqs, seqs_duration, \
        train_subset, val_subset
    from experiments.ch14_skel.b_preprocess import feat_seqs
    from experiments.ch14_skel.c_models import build_encoder

    # from sltools.ch14_bgr.a_data import tmpdir, gloss_seqs, seqs_duration, \
    #     train_subset, val_subset
    # from sltools.ch14_bgr.b_preprocess import feats_seqs
    # from sltools.ch14_bgr.c_models import build_encoder

    # Load data -------------------------------------------------------------------------

    feats_seqs_train = subset(feat_seqs, train_subset)
    gloss_seqs_train = subset(gloss_seqs, train_subset)
    seqs_durations_train = subset(seqs_duration, train_subset)

    feats_seqs_val = subset(feat_seqs, val_subset)
    gloss_seqs_val = subset(gloss_seqs, val_subset)
    seqs_durations_val = subset(seqs_duration, val_subset)

    # Build model -----------------------------------------------------------------------

    chains_lengths = [5] * 20
    n_states = sum(chains_lengths) + 1
    max_len = 128
    batch_size = 16
    posterior = PosteriorModel(build_encoder, n_states, max_len, batch_size)
    recognizer = HMMRecognizer(chains_lengths, posterior)

    # Run training iterations -----------------------------------------------------------

    report = shelve.open(os.path.join(tmpdir, "hmm_report"),
                         protocol=pkl.HIGHEST_PROTOCOL)
    resume_at = len(report) - 1
    l_rates = [0.01, 0.01] + [0.005] * 7 + [0.001] * 7
    epochs = [20, 20] + [7] * (len(l_rates) - 2)
    refit_posterior = [False, False] + [True] * (len(l_rates) - 2)

    for i in range(len(l_rates)):
        print("# It. {} -----------------------------------------------------".format(i))

        # Resume interrupted training
        if i < resume_at:
            print("Skipping")
            continue
        elif i == resume_at:
            print("reloading from partial training")
            recognizer = report[str(i)]['model']
            continue

        # Fit
        fit_args = {
            'refit': True,
            'priors_smoothing': .5,
            'hmm_fit_args': {
                'verbose': False, 'stop_threshold': 1e-3, 'max_iterations': 50},
            'posterior_fit_args': {
                'l_rate': l_rates[i], 'n_epochs': epochs[i], 'refit': refit_posterior[i],
                'loss': "hinge", 'updates': "adam",
                'weights': compute_weights, 'filter_chunks': filter_chunks,
                'callback': callback}
        }

        recognizer.fit(
            feats_seqs_train, gloss_seqs_train, **fit_args)

        # Update report
        if str(i - 1) in report.keys():
            previous_recognizer = report[str(i - 1)]['model']
        else:
            previous_recognizer = None
        train_report = epoch_perfs(
            recognizer,
            feats_seqs_train, gloss_seqs_train, seqs_durations_train,
            previous_recognizer)
        val_report = epoch_perfs(
            recognizer,
            feats_seqs_val, gloss_seqs_val, seqs_durations_val,
            previous_recognizer)
        report[str(i)] = {'train_report': train_report,
                          'val_report': val_report,
                          'model': pkl.loads(pkl.dumps(recognizer)),
                          'fit_args': fit_args,
                          'batch_losses': batch_losses,
                          'epoch_losses': epoch_losses}
        report.sync()
        batch_losses.clear()
        epoch_losses.clear()

        # Printout
        print("HMM Jaccard index: {:0.3f}".format(train_report['jaccard']))
        print("Framewise: {:0.3f}".format(train_report['framewise']))
        print("ANN Jaccard index: {:0.3f}".format(train_report['posterior_jaccard']))
        print("Framewise: {:0.3f}".format(train_report['posterior_framewise']))
        print("State-wise: {:0.3f}".format(train_report['statewise_framewise']))
        print("HMM Jaccard index: {:0.3f}".format(val_report['jaccard']))
        print("Framewise: {:0.3f}".format(val_report['framewise']))
        print("ANN Jaccard index: {:0.3f}".format(val_report['posterior_jaccard']))
        print("Framewise: {:0.3f}".format(val_report['posterior_framewise']))
        print("State-wise: {:0.3f}".format(val_report['statewise_framewise']))


if __name__ == "__main__":
    main()
