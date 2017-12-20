#!/usr/bin/env python3

import os
import pickle as pkl
import shelve
from functools import partial
import logging

import numpy as np
import lasagne
from lproc import rmap, subset

from sltools.utils import gloss2seq
from sltools.models.rnn import build_predict_fn, build_train_fn, seqs2batches
from sltools.nn_utils import compute_scores, seq_hinge_loss

from experiments.hmmvsrnn_reco.a_data import tmpdir, gloss_seqs, durations, \
    train_subset, val_subset, vocabulary


# Report setting ------------------------------------------------------------------------

experiment_name = "fusion_171219"
report = shelve.open(os.path.join(tmpdir, experiment_name),
                     protocol=pkl.HIGHEST_PROTOCOL)
with open(__file__) as f:
    this_script = f.read()
if "script" in report.keys():
    logging.warning("The script has changed since the previous time")
    report["script_altered"] = this_script
else:
    report["script"] = this_script


# Data ----------------------------------------------------------------------------------

from experiments.hmmvsrnn_reco.b_preprocess import skel_feat_seqs
from experiments.hmmvsrnn_reco.b_preprocess import bgr_feat_seqs

feats_seqs_train = [
    subset(skel_feat_seqs, train_subset),
    subset(bgr_feat_seqs, train_subset)
    ]
gloss_seqs_train = subset(gloss_seqs, train_subset)
seqs_durations_train = subset(durations, train_subset)
target_train = rmap(lambda g, d: gloss2seq(g, d, 0),
                    gloss_seqs_train, seqs_durations_train)

feats_seqs_val = [
    subset(skel_feat_seqs, val_subset),
    subset(bgr_feat_seqs, val_subset)
    ]
gloss_seqs_val = subset(gloss_seqs, val_subset)
seqs_durations_val = subset(durations, val_subset)
target_val = rmap(lambda g, d: gloss2seq(g, d, 0), 
                  gloss_seqs_val, seqs_durations_val)

# Model ---------------------------------------------------------------------------------

# from experiments.hmmvsrnn_reco.c_models import skel_lstm
# from experiments.hmmvsrnn_reco.c_models import bgr_lstm
from experiments.hmmvsrnn_reco.c_models import fusion_lstm

max_time = 128
batch_size = 16

# Skeleton end-to-end
# model_dict = skel_lstm(feats_shape=skel_feat_seqs[0][0].shape,
#                        batch_size=batch_size, max_time=max_time)
# BGR end-to-end
# layers_data = build_lstm(skel_feats_shape=skel_feat_seqs[0][0][0].shape,
#                          bgr_feats_shape=feat_seqs_train[0][1][0].shape,
#                          batch_size=batch_size, max_time=max_time)
# Fusion end-to-end
model_dict = fusion_lstm(skel_feats_shape=skel_feat_seqs[0][0].shape,
                         bgr_feats_shape=bgr_feat_seqs[0][0].shape,
                         batch_size=batch_size, max_time=max_time)

predict_fn = build_predict_fn(model_dict, batch_size, max_time)

# Training ------------------------------------------------------------------------------

weights = np.unique(np.concatenate(target_train), return_counts=True)[1] ** -0.7
weights *= 21 / weights.sum()
loss_fn = partial(seq_hinge_loss, delta=.5, weights=weights)
# loss_fn = partial(seq_ce_loss, weights=weights)
updates_fn = lasagne.updates.adam
save_every = 5
resume_at = sorted([int(e[6:]) for e in report.keys()
                    if e.startswith("epoch")
                    and "params" in report[e].keys()])
resume_at = -1 if len(resume_at) == 0 else resume_at[-1]
min_progress = 2e-3  # if improvement is below, decrease learning rate
l_rate = .001
train_batch_fn = build_train_fn(model_dict, max_time, model_dict['warmup'],
                                loss_fn, updates_fn)

batch_losses = []
multibatch_losses = np.array([])

for e in range(150):
    # Resume if possible ----------------------------------------------------------------

    if e < resume_at:
        print("\repoch {:>5d} : skipped".format(e))
        continue

    if e == resume_at:
        print("\repoch {:>5d} : resumed".format(e))
        epoch_report = report["epoch {:03d}".format(e)]
        min_progress *= epoch_report['l_rate'] / l_rate
        l_rate = epoch_report['l_rate']
        params = epoch_report['params']
        all_layers = lasagne.layers.get_all_layers(model_dict['l_linout'])
        lasagne.layers.set_all_param_values(all_layers, params)
        continue

    # Train one epoch -------------------------------------------------------------------

    batch_losses = []
    batch_iter = seqs2batches(feats_seqs_train, target_train,
                              batch_size, max_time, model_dict['warmup'],
                              shuffle=True, drop_last=True)
    
    for i, minibatch in enumerate(batch_iter):
        last_loss = float(train_batch_fn(*minibatch, l_rate))
        batch_losses.append(last_loss)
        if i % 30 == 0:
            print("\rbatch loss : {:>7.4f}".format(last_loss), end='', flush=True)

    # Generate report -------------------------------------------------------------------

    multibatch_losses = np.concatenate((multibatch_losses, np.asarray(batch_losses)))
    running_loss = batch_losses[0]
    for i in range(1, len(batch_losses)):
        running_loss = .99 * running_loss + .01 * batch_losses[i]
    print("\repoch {:>5d} loss : {:2.4f}       ".format(e + 1, running_loss))

    epoch_report = {
        'batch_losses': batch_losses,
        'epoch_loss': running_loss,
        'l_rate': l_rate
    }

    if (e + 1) % save_every == 0:
        predictions = [np.argmax(p, axis=1) for p in predict_fn(feats_seqs_train)]
        j, p, c = compute_scores(predictions, target_train, vocabulary)
        predictions = [np.argmax(p, axis=1) for p in predict_fn(feats_seqs_val)]
        jv, pv, cv = compute_scores(predictions, target_val, vocabulary)
        epoch_report['train_scores'] = \
            {'jaccard': j, 'framewise': p, 'confusion': c}
        epoch_report['val_scores'] = \
            {'jaccard': jv, 'framewise': pv, 'confusion': cv}

        print("scores: {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}".format(j, p, jv, pv))

        all_layers = lasagne.layers.get_all_layers(model_dict['l_linout'])
        params = lasagne.layers.get_all_param_values(all_layers)
        epoch_report['params'] = params

        # Update learning rate ----------------------------------------------------------

        b = np.cov(multibatch_losses, np.arange(len(multibatch_losses)))[1, 0] \
            / np.var(np.arange(len(multibatch_losses)))
        print('progress ~= {}'.format(b * len(multibatch_losses)))
        if b * len(multibatch_losses) > - min_progress:
            print("decreasing learning rate: {} -> {}".format(l_rate, l_rate * .3))
            l_rate *= .3
            min_progress *= .3
        multibatch_losses = np.array([])

    report["epoch {:03d}".format(e)] = epoch_report
    report.sync()
