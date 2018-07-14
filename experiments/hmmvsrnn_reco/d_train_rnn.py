#!/usr/bin/env python3

import os
import shelve
from functools import partial
import logging
import argparse
import datetime
import json
import time

import numpy as np
import lasagne
import seqtools
from lproc import rmap, subset

from sltools.utils import gloss2seq
from sltools.models.rnn import build_predict_fn, build_train_fn
from sltools.nn_utils import compute_scores, seq_ce_loss, adjust_length

from experiments.hmmvsrnn_reco.a_data import tmpdir, gloss_seqs, durations, \
    train_subset, val_subset, vocabulary


# Script arguments ----------------------------------------------------------------------

argparser = argparse.ArgumentParser()
argparser.add_argument("--name")
argparser.add_argument("--modality")
argparser.add_argument("--variant")
date = datetime.date.today()
argparser.add_argument("--date",
                       default="{:02d}{:02d}{:02d}".format(
                           date.year % 100, date.month, date.day))
argparser.add_argument("--notes", default="")
argparser.add_argument("--batch_size", type=int)
argparser.add_argument("--max_time", type=int)
argparser.add_argument("--encoder_kwargs")
args = argparser.parse_args()

experiment_name = args.name
modality = args.modality
variant = args.variant
date = args.date
notes = args.notes
batch_size = args.batch_size
max_time = args.max_time
encoder_kwargs = json.loads(args.encoder_kwargs)


# Report setting ------------------------------------------------------------------------

report = shelve.open(os.path.join(tmpdir, experiment_name), protocol=-1)

report['meta'] = {
    'model': "rnn",
    'modality': modality,
    'variant': variant,
    'date': date,
    'notes': notes,
    'experiment_name': experiment_name
}

report['args'] = {
    'batch_size': batch_size,
    'max_time': max_time,
    'encoder_kwargs': encoder_kwargs
}


with open(__file__) as f:
    this_script = f.read()
if "script" in report.keys():
    if report["script"] != this_script:
        logging.warning("The script has changed since the previous time")
        report["script_altered"] = this_script
    else:
        report["script"] = this_script


# Data ----------------------------------------------------------------------------------

if modality == "skel":
    from experiments.hmmvsrnn_reco.b_preprocess import skel_feat_seqs
    feat_seqs = [skel_feat_seqs]
elif modality == "bgr":
    from experiments.hmmvsrnn_reco.b_preprocess import bgr_feat_seqs
    feat_seqs = [bgr_feat_seqs]
elif modality == "fusion":
    from experiments.hmmvsrnn_reco.b_preprocess import skel_feat_seqs
    from experiments.hmmvsrnn_reco.b_preprocess import bgr_feat_seqs
    feat_seqs = [skel_feat_seqs, bgr_feat_seqs]
elif modality == "transfer":
    from experiments.hmmvsrnn_reco.b_preprocess import transfer_feat_seqs
    feat_seqs = transfer_feat_seqs(encoder_kwargs['transfer_from'],
                                   encoder_kwargs['freeze_at'])
else:
    raise ValueError()

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

del feat_seqs, durations, gloss_seqs  # fool proofing


# Model ---------------------------------------------------------------------------------

if modality == "skel":  # Skeleton end-to-end
    from experiments.hmmvsrnn_reco.c_models import skel_lstm as build_model_fn
elif modality == "bgr":  # BGR end-to-end
    from experiments.hmmvsrnn_reco.c_models import bgr_lstm as build_model_fn
elif modality == "fusion":  # Fusion end-to-end
    from experiments.hmmvsrnn_reco.c_models import fusion_lstm as build_model_fn
elif modality == "transfer":  # Skeleton transfer
    from experiments.hmmvsrnn_reco.c_models import transfer_lstm as build_model_fn
else:
    raise ValueError

model_dict = build_model_fn(
    *[f[0][0].shape for f in feat_seqs_train],
    batch_size=batch_size, max_time=max_time,
    encoder_kwargs=encoder_kwargs)

predict_fn = build_predict_fn(model_dict, batch_size, max_time)


# Training ------------------------------------------------------------------------------

if modality == 'transfer':
    weights = np.unique(np.concatenate(targets_train), return_counts=True)[1] ** -0.2
else:
    weights = np.unique(np.concatenate(targets_train), return_counts=True)[1] ** -0.7
weights *= 21 / weights.sum()
loss_fn = partial(seq_ce_loss, weights=weights)
updates_fn = lasagne.updates.adam

train_batch_fn = build_train_fn(
    model_dict, max_time, model_dict['warmup'],
    loss_fn, updates_fn)

if modality == 'transfer':  # slow down learning otherwise best epoch is skipped
    save_every = 1
    l_rate = 1e-3
else:
    save_every = 5
    l_rate = 1e-3

min_progress = 1e-3  # if improvement is below, decrease learning rate

e = 0


def resume(report_prefix):
    global min_progress, l_rate, e

    previous = sorted([int(r[len(report_prefix) + 1:]) for r in report.keys()
                       if r.startswith(report_prefix)
                       and "params" in report[r].keys()])

    if len(previous) > 0:
        resume_at = previous[-1]
        epoch_report = report["{} {:04d}".format(report_prefix, resume_at)]
        print("\r{} {:>4d} : resumed".format(report_prefix, resume_at))

        min_progress *= epoch_report['l_rate'] / l_rate
        l_rate = epoch_report['l_rate']
        params = epoch_report['params']
        all_layers = lasagne.layers.get_all_layers(model_dict['l_linout'])
        lasagne.layers.set_all_param_values(all_layers, params)
        e = resume_at + 1


def train_one_epoch(report_key):
    batch_losses = []
    step = max_time - 2 * model_dict['warmup']

    # turn sequences into chunks
    chunks = [(i, k, min(d, k + max_time))
              for i, d in enumerate(durations_train)
              for k in range(0, d - model_dict['warmup'], step)]
    chunked_sequences = []
    for feat in feat_seqs_train:
        def get_chunk(i, t1, t2, feat_=feat):
            return adjust_length(feat_[i][t1:t2], size=max_time, pad=0)

        chunked_sequences.append(seqtools.starmap(get_chunk, chunks))
    chunked_sequences.append(seqtools.starmap(
        lambda i, t1, t2: adjust_length(targets_train[i][t1:t2], max_time, pad=-1),
        chunks))
    chunked_sequences.append([np.int32(t2 - t1) for _, t1, t2 in chunks])
    chunked_sequences = seqtools.collate(chunked_sequences)

    perm = np.random.permutation(len(chunked_sequences))
    chunked_sequences = seqtools.gather(chunked_sequences, perm)

    # turn into minibatches
    null_sample = chunked_sequences[0]
    n_features = len(null_sample)

    def collate(b_):
        return [np.array([b_[i][c] for i in range(batch_size)])
                for c in range(n_features)]

    # todo: shuffle
    minibatches = seqtools.batch(
        chunked_sequences, batch_size, drop_last=True,
        collate_fn=collate)
    minibatches = seqtools.prefetch(
        minibatches, max_cached=10, nworkers=2)

    # train
    t = time.time()
    running_loss = None
    for b in minibatches:
        loss = train_batch_fn(*b, l_rate)
        batch_losses.append(loss)
        running_loss = .99 * running_loss + .01 * loss if running_loss else loss
        if time.time() - t > 1:
            print("\rbatch loss : {:>2.4f}".format(running_loss), end='', flush=True)
            t = time.time()

    print()

    # report
    report[report_key] = {
        'batch_losses': batch_losses,
        'epoch_loss': running_loss,
        'l_rate': l_rate
    }


def extra_report(report_key):
    epoch_report = report[report_key]
    predictions = [np.argmax(p, axis=1) for p in predict_fn(feat_seqs_train)]
    j, p, c = compute_scores(predictions, targets_train, vocabulary)
    predictions = [np.argmax(p, axis=1) for p in predict_fn(feat_seqs_val)]
    jv, pv, cv = compute_scores(predictions, targets_val, vocabulary)
    epoch_report['train_scores'] = \
        {'jaccard': j, 'framewise': p, 'confusion': c}
    epoch_report['val_scores'] = \
        {'jaccard': jv, 'framewise': pv, 'confusion': cv}

    print("    scores: {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}".format(j, p, jv, pv))

    all_layers = lasagne.layers.get_all_layers(model_dict['l_linout'])
    params = lasagne.layers.get_all_param_values(all_layers)
    epoch_report['params'] = params

    report[report_key] = epoch_report


def update_setup(epoch_prefix):
    global l_rate, min_progress

    # Compute average loss progress by epoch over the last 10 epochs (linear reg)
    last_reports = sorted(k for k in report.keys() if k.startswith(epoch_prefix))[-10:]
    multibatch_losses = np.concatenate([report[k]["batch_losses"] for k in last_reports])
    avg_progress = np.cov(multibatch_losses, np.arange(len(multibatch_losses)))[1, 0] \
        / np.var(np.arange(len(multibatch_losses))) \
        * len(multibatch_losses) / len(last_reports)

    print('   progress ~ {:.4e}'.format(avg_progress))
    if avg_progress > - min_progress:
        print("decreasing learning rate: {} -> {}".format(l_rate, l_rate * .3))
        l_rate *= .3
        min_progress *= .3


resume("epoch")

while e < (50 if modality == 'transfer' else 150):
    train_one_epoch("epoch {:04d}".format(e))
    if (e + 1) % save_every == 0:
        extra_report("epoch {:04d}".format(e))
    update_setup("epoch")
    e += 1
    report.sync()
