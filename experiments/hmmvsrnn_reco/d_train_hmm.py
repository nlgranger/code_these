#!/usr/bin/env python3

import os
import logging
import pickle as pkl
import shelve
from copy import deepcopy
from functools import partial
import numpy as np
from lproc import rmap, subset

from sltools.utils import gloss2seq
from sltools.models import HMMRecognizer, PosteriorModel, hmm_perfs

from experiments.hmmvsrnn_reco.a_data import tmpdir, gloss_seqs, durations, \
    train_subset, val_subset, vocabulary


# Report setting ------------------------------------------------------------------------

model = "hmm"
modality = "skel"
variant = "tc7"
date = "180215"

experiment_name = model + "_" + modality + "_" \
    + (variant + "_" if variant != "" else "") + date
report = shelve.open(os.path.join(tmpdir, experiment_name),
                     protocol=pkl.HIGHEST_PROTOCOL)
with open(__file__) as f:
    this_script = f.read()
if "script" in report.keys():
    if report["script"] != this_script:
        logging.warning("The script has changed since the previous time")
        report["script_altered"] = this_script
    else:
        report["script"] = this_script


# Data ----------------------------------------------------------------------------------

if modality == "skel":  # Skeleton end-to-end
    from experiments.hmmvsrnn_reco.b_preprocess import skel_feat_seqs
    feat_seqs = [skel_feat_seqs]
elif modality == "bgr":  # BGR end-to-end
    from experiments.hmmvsrnn_reco.b_preprocess import bgr_feat_seqs
    feat_seqs = [bgr_feat_seqs]
elif modality == "fusion":  # Fusion end-to-end
    from experiments.hmmvsrnn_reco.b_preprocess import skel_feat_seqs
    from experiments.hmmvsrnn_reco.b_preprocess import bgr_feat_seqs
    feat_seqs = [skel_feat_seqs, bgr_feat_seqs]
elif modality == "transfer":  # Transfer
    from experiments.hmmvsrnn_reco.b_preprocess_transfer import transfer_features
    feat_seqs = [transfer_features(
        "rnn_skel_180201", "rnn_skel",
        max_time=128, batch_size=16,
        encoder_kwargs={'tconv_sz': 17, 'filter_dilation': 1})]
else:
    raise ValueError

feat_seqs_train = [subset(f, train_subset) for f in feat_seqs]
gloss_seqs_train = subset(gloss_seqs, train_subset)
durations_train = subset(durations, train_subset)
targets_train = rmap(lambda g, d: gloss2seq(g, d, 0),
                     gloss_seqs_train, durations_train)
feat_seqs_val = [subset(f, val_subset) for f in feat_seqs]
gloss_seqs_val = subset(gloss_seqs, val_subset)
durations_val = subset(durations, val_subset)
targets_val = rmap(lambda g, d: gloss2seq(g, d, 0),
                    gloss_seqs_val, durations_val)

# Model ---------------------------------------------------------------------------------

chains_lengths = [5] * 20

if modality == "skel":
    from experiments.hmmvsrnn_reco.c_models import skel_encoder as encoder
    encoder_kwargs = {'tconv_sz': 7, 'filter_dilation': 1}
    posterior_kwargs = {'max_len': 128, 'batch_size': 16}
elif modality == "bgr":
    from experiments.hmmvsrnn_reco.c_models import bgr_encoder as encoder
    encoder_kwargs = {'tconv_sz': 17, 'filter_dilation': 1}
    posterior_kwargs = {'max_len': 128, 'batch_size': 16}
elif modality == "fusion":
    from experiments.hmmvsrnn_reco.c_models import fusion_encoder as encoder
    encoder_kwargs = {'tconv_sz': 17, 'filter_dilation': 1}
    posterior_kwargs = {'max_len': 128, 'batch_size': 16}
elif modality == "transfer":
    from experiments.hmmvsrnn_reco.c_models import identity_encoder as encoder
    encoder_kwargs = {'warmup': (17 * 1) // 2}
    posterior_kwargs = {'max_len': 128, 'batch_size': 16}
else:
    raise ValueError

encoder = partial(encoder, **encoder_kwargs)
posterior = PosteriorModel(encoder, sum(chains_lengths) + 1, **posterior_kwargs)
recognizer = HMMRecognizer(chains_lengths, posterior, vocabulary)

report['chains_lengths'] = chains_lengths
report['encoder_kwargs'] = encoder_kwargs
report['posterior_kwargs'] = posterior_kwargs


# Run training iterations ---------------------------------------------------------------

resume_at = sorted([int(e[6:]) for e in report.keys()
                    if e.startswith("epoch")
                    and "params" in report[e].keys()])
resume_at = -1 if len(resume_at) == 0 else resume_at[-1]

# Posterior training settings
l_rate = .001
updates = 'adam'
loss = 'cross_entropy'
weight_smoothing = .7
min_progress = .01
epoch_schedule = [20, 20] + [7] * 14  # <<<< TODO
refit_schedule = [False, False] + [True] * (len(epoch_schedule) - 2)

# prior training_settings
prior_smoothing = 0  # Ignoring priors!!!!

for i in range(len(epoch_schedule)):
    print("# It. {} -----------------------------------------------------".format(i))

    # Resume interrupted training
    if i < resume_at:
        print("Skipping")
        continue
    elif i == resume_at:
        print("reloading from partial training")
        recognizer = report[str(i)]['model']
        fit_settings = report[str(i)]['fit_settings']
        chains_lengths = fit_settings['chains_lengths']
        l_rate = fit_settings['l_rate']
        updates = fit_settings['updates']
        loss = fit_settings['loss']
        min_progress = fit_settings['min_progress']
        epoch_schedule = fit_settings['epoch_schedule']
        refit_schedule = fit_settings['refit_schedule']
        prior_smoothing = fit_settings['prior_smoothing']
        continue

    # Fit posterior
    state_assignment = recognizer.align_states(feat_seqs_train, gloss_seqs_train,
                                               linear=(i == 0))

    counts = np.unique(np.concatenate(state_assignment), return_counts=True)[1]
    weights = np.power((counts + 100) / counts.max(), - weight_smoothing)
    weights = weights / np.sum(weights) * recognizer.nstates

    batch_losses = []
    epoch_losses = []

    def callback(epoch, n_epochs, batch, n_batches, batch_loss):
        batch_losses.append(batch_loss)
        print("\rbatch {:>5d}/{:<5d} batch_loss : {:2.4f}".format(
            batch + 1, n_batches, batch_loss), end='', flush=True)

        if batch + 1 == n_batches:
            epoch_loss = np.mean(batch_losses[-n_batches // 10:])
            epoch_losses.append(epoch_loss)
            print("\repoch {:>5d}/{:<5d} batch_loss : {:2.4f}".format(
                epoch + 1, n_epochs, epoch_loss))

    recognizer.fit_posterior(feat_seqs_train, state_assignment,
                             refit=refit_schedule[i],
                             weights=weights, n_epochs=epoch_schedule[i],
                             l_rate=l_rate, loss=loss, updates=updates,
                             callback=callback)

    # Fit priors
    recognizer.fit_priors(feat_seqs_train, gloss_seqs_train,
                          smoothing=prior_smoothing)

    # Fit transitions
    recognizer.fit_transitions(feat_seqs_train, gloss_seqs_train,
                               stop_threshold=1e-3, max_iterations=50,
                               verbose=False)

    # Update report
    if str(i - 1) in report.keys():
        previous_recognizer = report[str(i - 1)]['model']
    else:
        previous_recognizer = None
    train_report = hmm_perfs(
        recognizer,
        feat_seqs_train, gloss_seqs_train, durations_train,
        vocabulary, previous_recognizer)
    val_report = hmm_perfs(
        recognizer,
        feat_seqs_val, gloss_seqs_val, durations_val,
        vocabulary, previous_recognizer)
    report["epoch {:>3d}".format(i)] = {
        'train_scores': train_report,
        'val_scores': val_report,
        'model': deepcopy(recognizer),
        'fit_settings': {
            'l_rate': .001,
            'updates': updates,
            'loss': loss,
            'weight_smoothing': weight_smoothing,
            'min_progress': min_progress,
            'epoch_schedule': epoch_schedule,
            'refit_schedule': refit_schedule,
            'prior_smoothing': prior_smoothing,
        },
        'batch_losses': batch_losses,
        'epoch_losses': epoch_losses}
    report.sync()

    # Update training settings
    n_steps = len(batch_losses)
    b = np.cov(batch_losses, np.arange(n_steps))[1, 0] \
        / np.var(np.arange(n_steps))
    if b * n_steps > - min_progress:
        l_rate *= .3
        min_progress *= .3

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
