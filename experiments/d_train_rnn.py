#!/usr/bin/env python3

import os
import shelve
from functools import partial
import logging
import argparse
import datetime
import json
import time
import itertools

import numpy as np
import lasagne
import seqtools
from lproc import rmap, subset

from sltools.utils import gloss2seq
from sltools.models.rnn import build_predict_fn, build_train_fn
from sltools.nn_utils import compute_scores, seq_ce_loss, adjust_length

from experiments.a_data import cachedir, gloss_seqs, durations, \
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
argparser.add_argument("--mono", action='store_const', const=True, default=False)
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

report = shelve.open(os.path.join(cachedir, experiment_name), protocol=-1)

report['meta'] = {
    'model': "rnn" + ("_mono" if args.mono else ""),
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
    from experiments.b_preprocess import skel_feat_seqs
    feat_seqs = [skel_feat_seqs]
elif modality == "bgr":
    from experiments.b_preprocess import bgr_feat_seqs
    feat_seqs = [bgr_feat_seqs]
elif modality == "fusion":
    from experiments.b_preprocess import skel_feat_seqs
    from experiments.b_preprocess import bgr_feat_seqs
    feat_seqs = [skel_feat_seqs, bgr_feat_seqs]
elif modality == "transfer":
    from experiments.b_preprocess import transfer_feat_seqs
    feat_seqs = transfer_feat_seqs(encoder_kwargs['transfer_from'],
                                   encoder_kwargs['freeze_at'])
else:
    raise ValueError()

feat_seqs_train = [subset(f, train_subset) for f in feat_seqs]
gloss_seqs_train = subset(gloss_seqs, train_subset)
durations_train = durations[train_subset]
targets_train = rmap(lambda g, d: gloss2seq(g, d, 0),
                     gloss_seqs_train, durations_train)
feat_seqs_val = [subset(f, val_subset) for f in feat_seqs]
gloss_seqs_val = subset(gloss_seqs, val_subset)
durations_val = durations[val_subset]
targets_val = rmap(lambda g, d: gloss2seq(g, d, 0),
                   gloss_seqs_val, durations_val)

del feat_seqs, durations, gloss_seqs  # fool proofing


# Model ---------------------------------------------------------------------------------

if modality == "skel" and not args.mono:  # Skeleton end-to-end
    from experiments.c_models import skel_lstm as build_model_fn
elif modality == "skel" and args.mono:  # Skeleton end-to-end
    from experiments.c_models import mono_lstm as build_model_fn
elif modality == "bgr":  # BGR end-to-end
    from experiments.c_models import bgr_lstm as build_model_fn
elif modality == "fusion":  # Fusion end-to-end
    from experiments.c_models import fusion_lstm as build_model_fn
elif modality == "transfer":  # Skeleton transfer
    from experiments.c_models import transfer_lstm as build_model_fn
else:
    raise ValueError

model_dict = build_model_fn(
    *[f[0][0].shape for f in feat_seqs_train],
    batch_size=batch_size, max_time=max_time,
    encoder_kwargs=encoder_kwargs)

predict_fn = build_predict_fn(model_dict, batch_size, max_time)
# predict_fn = None


# Training ------------------------------------------------------------------------------

alpha = 0.2 if modality == 'transfer' else 0.7
counts = np.unique(np.concatenate(targets_train), return_counts=True)[1]
weights = 1 / (21 * (counts / np.sum(counts)) ** alpha)
loss_fn = partial(seq_ce_loss, weights=weights)
updates_fn = lasagne.updates.adam

train_batch_fn = build_train_fn(
    model_dict, max_time, model_dict['warmup'],
    loss_fn, updates_fn)
# train_batch_fn = None

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


# turn sequences into chunks
class MinibatchSampler:
    def __init__(self):
        step = max_time - 2 * model_dict['warmup']
        self.chunks = np.array([(i, k, min(d, k + max_time))
                                for i, d in enumerate(durations_train)
                                for k in range(0, d - model_dict['warmup'], step)])
        self.weights = np.ones((len(self.chunks),)) / len(self.chunks)

    def __call__(self):
        # sample chunks to cover all timesteps
        batch_idx = np.random.choice(
            len(self.chunks), size=batch_size, replace=False, p=self.weights)
        self.weights[batch_idx] /= 10
        self.weights /= np.sum(self.weights)
        batch_chunks = np.copy(self.chunks[batch_idx])

        # randomize subsegments edges
        batch_dur = durations_train[batch_chunks[:, 0]]
        offset = np.random.randint(-max_time // 2, max_time // 2, size=batch_size)
        offset = np.fmax(0, batch_chunks[:, 1] + offset) - batch_chunks[:, 1]
        offset = np.fmin(batch_dur, batch_chunks[:, 2] + offset) - batch_chunks[:, 2]
        batch_chunks[:, (1, 2)] += offset[:, None]

        # assemble minibatch
        out = []
        for modality_seqs in feat_seqs_train:
            out.append(np.stack([adjust_length(modality_seqs[i][a:b], max_time)
                                 for i, a, b in batch_chunks]))

        out.append(np.stack([adjust_length(targets_train[i][a:b], max_time)
                             for i, a, b in batch_chunks]).astype(np.int32))
        out.append(batch_dur.astype(np.int32))

        return tuple(out)


def train_n_steps(report_key, batch_it, n_steps):
    batch_losses = []

    # train
    running_loss = None
    t = time.time()
    for minibatch in itertools.islice(batch_it, n_steps):
        loss = train_batch_fn(*minibatch, l_rate)
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
    last_reports = sorted(k for k in report.keys() if k.startswith(epoch_prefix))[-5:]
    multibatch_losses = np.concatenate([report[k]["batch_losses"] for k in last_reports])
    avg_progress = np.cov(multibatch_losses, np.arange(len(multibatch_losses)))[1, 0] \
        / np.var(np.arange(len(multibatch_losses))) \
        * len(multibatch_losses) / len(last_reports)

    print('    progress ~ {:.4e}'.format(avg_progress))
    if avg_progress > - min_progress and l_rate > 1e-4:
        print("decreasing learning rate: {} -> {}".format(l_rate, l_rate * .3))
        l_rate /= 3
        min_progress /= 3


resume("epoch")

minibatch_it = seqtools.load_buffers(
    MinibatchSampler(), max_cached=10, nworkers=1, start_hook=np.random.seed)
step = max_time - 2 * model_dict['warmup']
steps_by_epoch = int(sum(durations_train) // (step * batch_size))
while e < (50 if modality == 'transfer' else 150):
    train_n_steps("epoch {:04d}".format(e), minibatch_it, steps_by_epoch)
    if (e + 1) % save_every == 0:
        extra_report("epoch {:04d}".format(e))
    update_setup("epoch")
    e += 1
    report.sync()
