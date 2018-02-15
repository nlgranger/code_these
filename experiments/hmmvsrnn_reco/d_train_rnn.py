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
from sltools.nn_utils import compute_scores, seq_ce_loss, seq_hinge_loss

from experiments.hmmvsrnn_reco.a_data import tmpdir, gloss_seqs, durations, \
    train_subset, val_subset, vocabulary


# Report setting ------------------------------------------------------------------------

model = "rnn"
modality = "skel"
variant = "tc7"
date = "180215"

experiment_name = model + "_" + modality + "_" \
    + (variant + "_" if variant != "" else "") + date
report = shelve.open(os.path.join(tmpdir, experiment_name),
                     protocol=pkl.HIGHEST_PROTOCOL)
with open(__file__) as f:
    this_script = f.read()
if "script" in report.keys() and this_script != report["script"]:
    logging.warning("The script has changed since the previous time")
    report["script_altered"] = this_script
else:
    report["script"] = this_script


# Data ----------------------------------------------------------------------------------

if modality == "skel":  # Skeleton end-to-end
    from experiments.hmmvsrnn_reco.b_preprocess import skel_feat_seqs
    feats_seqs_train = [subset(skel_feat_seqs, train_subset)]
    feats_seqs_val = [subset(skel_feat_seqs, val_subset)]
elif modality == "bgr":  # BGR end-to-end
    from experiments.hmmvsrnn_reco.b_preprocess import bgr_feat_seqs
    feats_seqs_train = [subset(bgr_feat_seqs, train_subset)]
    feats_seqs_val = [subset(bgr_feat_seqs, val_subset)]
elif modality == "fusion":  # Fusion end-to-end
    from experiments.hmmvsrnn_reco.b_preprocess import skel_feat_seqs
    from experiments.hmmvsrnn_reco.b_preprocess import bgr_feat_seqs
    feats_seqs_train = [
        subset(skel_feat_seqs, train_subset),
        subset(bgr_feat_seqs, train_subset)
        ]
    feats_seqs_val = [
        subset(skel_feat_seqs, val_subset),
        subset(bgr_feat_seqs, val_subset)
        ]
elif modality == "transfer":  # Transfer
    from experiments.hmmvsrnn_reco.b_preprocess_transfer import transfer_features
    transfered_feats_seqs = transfer_features(
        "hmm_skel_180122", "hmm_skel",
        max_time=128, batch_size=16,
        encoder_kwargs={'tconv_sz': 17, 'filter_dilation': 1})
    feats_seqs_train = [subset(transfered_feats_seqs, train_subset)]
    feats_seqs_val = [subset(transfered_feats_seqs, val_subset)]

else:
    raise ValueError

# Annotations
gloss_seqs_train = subset(gloss_seqs, train_subset)
durations_train = subset(durations, train_subset)
targets_train = rmap(lambda g, d: gloss2seq(g, d, 0),
                     gloss_seqs_train, durations_train)

gloss_seqs_val = subset(gloss_seqs, val_subset)
durations_val = subset(durations, val_subset)
targets_val = rmap(lambda g, d: gloss2seq(g, d, 0),
                   gloss_seqs_val, durations_val)

# Model ---------------------------------------------------------------------------------

if modality == "skel":  # Skeleton end-to-end
    from experiments.hmmvsrnn_reco.c_models import skel_lstm
    model_args = {
        'batch_size': 16,
        'max_time': 128,
        'encoder_kwargs': {'tconv_sz': 7, 'filter_dilation': 1, 'dropout': .1}
    }
    model_dict = skel_lstm(feats_shape=skel_feat_seqs[0][0].shape, **model_args)
elif modality == "bgr":  # BGR end-to-end
    from experiments.hmmvsrnn_reco.c_models import bgr_lstm
    model_args = {
        'batch_size': 12,
        'max_time': 128,
        'encoder_kwargs': {}
    }
    model_dict = bgr_lstm(feats_shape=bgr_feat_seqs[0][0].shape, **model_args)
elif modality == "fusion":  # Fusion end-to-end
    from experiments.hmmvsrnn_reco.c_models import fusion_lstm
    model_args = {
        'batch_size': 12,
        'max_time': 128,
        'encoder_kwargs': {}
    }
    model_dict = fusion_lstm(skel_feats_shape=skel_feat_seqs[0][0].shape,
                             bgr_feats_shape=bgr_feat_seqs[0][0].shape, **model_args)
elif modality == "transfer":  # Skeleton transfer
    from experiments.hmmvsrnn_reco.c_models import transfer_lstm
    model_args = {
        'batch_size': 16,
        'max_time': 128,
        'warmup': (17 * 1) // 2
    }
    model_dict = transfer_lstm(
        feats_shape=transfered_feats_seqs[0][0].shape, **model_args)
else:
    raise ValueError

report['model_args'] = model_args

# Predictor function
predict_fn = build_predict_fn(
    model_dict, model_args['batch_size'], model_args['max_time'])

# Training ------------------------------------------------------------------------------

weights = np.unique(np.concatenate(targets_train), return_counts=True)[1] ** -0.7
weights *= 21 / weights.sum()
loss_fn = partial(seq_ce_loss, weights=weights)
updates_fn = lasagne.updates.adam
save_every = 5
resume_at = sorted([int(e[6:]) for e in report.keys()
                    if e.startswith("epoch")
                    and "params" in report[e].keys()])
resume_at = -1 if len(resume_at) == 0 else resume_at[-1]
min_progress = 1e-3  # if improvement is below, decrease learning rate
l_rate = 1e-3
train_batch_fn = build_train_fn(
    model_dict, model_args['max_time'], model_dict['warmup'], loss_fn, updates_fn)

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
    batch_iter = seqs2batches(
        feats_seqs_train, targets_train,
        model_args['batch_size'], model_args['max_time'], model_dict['warmup'],
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
        j, p, c = compute_scores(predictions, targets_train, vocabulary)
        predictions = [np.argmax(p, axis=1) for p in predict_fn(feats_seqs_val)]
        jv, pv, cv = compute_scores(predictions, targets_val, vocabulary)
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


# fine tune -----------------------------------------------------------------------------

best_epoch = sorted([(r['val_scores']['jaccard'], e)
                     for e, r in report.items()
                     if e.startswith("epoch")
                     and "params" in r.keys()])[-1][1]
print("fine-tuning from: {}".format(best_epoch))
epoch_report = report[best_epoch]
lasagne.layers.set_all_param_values(
    model_dict['l_linout'],
    epoch_report['params'])

weights = np.unique(np.concatenate(targets_train), return_counts=True)[1] ** -0.7
weights *= 21 / weights.sum()
loss_fn = partial(seq_hinge_loss, delta=.3, weights=weights)
updates_fn = lasagne.updates.adam
save_every = 5
resume_at = sorted([int(e[9:]) for e in report.keys()
                    if e.startswith("finetune")
                    and "params" in report[e].keys()])
resume_at = -1 if len(resume_at) == 0 else resume_at[-1]
l_rate = 1e-5
min_progress = 1e-5
train_batch_fn = build_train_fn(
    model_dict, model_args['max_time'], model_dict['warmup'], loss_fn, updates_fn)

multibatch_losses = np.array([])
batch_losses = []

for e in range(20):
    # Resume if possible --------------------------------------------------------------

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

    # Train one epoch -----------------------------------------------------------------

    batch_losses = []
    batch_iter = seqs2batches(
        feats_seqs_train, targets_train,
        model_args['batch_size'], model_args['max_time'], model_dict['warmup'],
        shuffle=True, drop_last=True)

    for i, minibatch in enumerate(batch_iter):
        last_loss = float(train_batch_fn(*minibatch, l_rate))
        batch_losses.append(last_loss)
        if i % 30 == 0:
            print("\rbatch loss : {:>7.4f}".format(last_loss), end='', flush=True)

    # Generate report -----------------------------------------------------------------

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
        j, p, c = compute_scores(predictions, targets_train, vocabulary)
        predictions = [np.argmax(p, axis=1) for p in predict_fn(feats_seqs_val)]
        jv, pv, cv = compute_scores(predictions, targets_val, vocabulary)
        epoch_report['train_scores'] = \
            {'jaccard': j, 'framewise': p, 'confusion': c}
        epoch_report['val_scores'] = \
            {'jaccard': jv, 'framewise': pv, 'confusion': cv}

        print("scores: {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}".format(j, p, jv, pv))

        all_layers = lasagne.layers.get_all_layers(model_dict['l_linout'])
        params = lasagne.layers.get_all_param_values(all_layers)
        epoch_report['params'] = params

        # Update learning rate --------------------------------------------------------

        b = np.cov(multibatch_losses, np.arange(len(multibatch_losses)))[1, 0] \
            / np.var(np.arange(len(multibatch_losses)))
        print('progress ~= {}'.format(b * len(multibatch_losses)))
        if b * len(multibatch_losses) > - min_progress:
            print("decreasing learning rate: {} -> {}".format(l_rate, l_rate * .3))
            l_rate *= .3
            min_progress *= .3
        multibatch_losses = np.array([])

    report["finetune {:03d}".format(e)] = epoch_report
    report.sync()
