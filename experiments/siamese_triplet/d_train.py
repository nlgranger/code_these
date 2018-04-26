import os
import json
import datetime
import argparse
import shelve
from functools import partial
import numpy as np
import theano
import theano.tensor as T
import lasagne
import seqtools

from experiments.siamese_triplet.a_data import cachedir, \
    durations, labels, recordings, \
    train_subset
from experiments.siamese_triplet.b_preprocess import skel_feat_seqs
from experiments.siamese_triplet.c_model import skel_rnn, build_predict_fn
from experiments.siamese_triplet.common import sample_episode, sample_triplets, \
    kernel_loss, triplet_loss, episode2minibatch, triplets2minibatches, \
    evaluate_matching


# ---------------------------------------------------------------------------------------

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
argparser.add_argument("--shots", type=int)
argparser.add_argument("--loss")
argparser.add_argument("--ep_voca_size", type=int)
argparser.add_argument("--encoder_kwargs")
args = argparser.parse_args()

experiment_name = args.name
modality = args.modality
variant = args.variant
date = args.date
notes = args.notes
batch_size = args.batch_size
max_time = args.max_time
shots = args.shots
ep_voca_size = args.ep_voca_size
encoder_kwargs = json.loads(args.encoder_kwargs)

report = shelve.open(os.path.join(cachedir, "rnn_report"))
report.clear()

# Data ------------------------------------------------------------------------------

feat_seqs_train = [seqtools.gather(skel_feat_seqs, train_subset)]
labels_train = labels[train_subset].astype(np.int32)
recordings_train = recordings[train_subset]
durations_train = durations[train_subset].astype(np.int32)
vocabulary = np.unique(labels_train)

del recordings, labels, durations, skel_feat_seqs

# Model -----------------------------------------------------------------------------

model_dict = skel_rnn(
    *tuple(f[0][0].shape for f in feat_seqs_train),
    batch_size=batch_size, max_time=max_time,
    encoder_kwargs=encoder_kwargs)

l_linout = model_dict['l_linout']
l_in = model_dict['l_in']
l_duration = model_dict['l_duration']

report['meta'] = {
    'date': date,
    'notes': notes,
    'modality': modality,
    'variant': variant,
    'max_time': max_time,
    'batch_size': batch_size,
    'shots': shots,
    'ep_voca_size': ep_voca_size,
    'encoder_kwargs': encoder_kwargs
}

# Training routines -----------------------------------------------------------------

if args.loss == "triplet":
    # update function
    l_rate_var = T.scalar('l_rate')
    linout = lasagne.layers.get_output(l_linout, deterministic=False)

    train_loss = triplet_loss(linout[:batch_size - batch_size % 3],
                              metric='squared', delta=0.7).mean()

    params = lasagne.layers.get_all_params(l_linout, trainable=True)
    updates = lasagne.updates.adam(
        train_loss, params, learning_rate=l_rate_var)

    update_fn = theano.function(
        [l.input_var for l in l_in] + [l_duration.input_var, l_rate_var],
        outputs=train_loss, updates=updates)

    # Minibatches
    triplets = sample_triplets(labels_train, recordings_train, vocabulary, 200000)

    train_minibatches = triplets2minibatches(feat_seqs_train, durations_train, triplets,
                                             batch_size, max_time)

elif args.loss == "kernel":
    # update function
    l_rate_var = T.scalar('l_rate')
    targets_var = T.ivector('labels')
    linout = lasagne.layers.get_output(l_linout, deterministic=False)

    train_loss = kernel_loss(linout, targets_var, ep_voca_size, shots).mean()

    params = lasagne.layers.get_all_params(l_linout, trainable=True)
    updates = lasagne.updates.adam(
        train_loss, params, learning_rate=l_rate_var)

    update_fn = theano.function(
        [l.input_var for l in l_in] + [l_duration.input_var, targets_var, l_rate_var],
        outputs=train_loss, updates=updates)

    # Minibatches
    episodes = [sample_episode(
                    labels_train, recordings_train, vocabulary,
                    ep_voca_size, shots, batch_size - ep_voca_size * shots)
                for _ in range(10000)]

    converter = partial(
        episode2minibatch,
        feat_seqs=feat_seqs_train, durations=durations_train, max_time=max_time)

    train_minibatches = seqtools.smap(converter, episodes)

else:
    raise ValueError("unsupported loss argument")

predict_fn = build_predict_fn(model_dict, report['meta']['batch_size'],
                              report['meta']['max_time'])


# Training iterations -------------------------------------------------------------------
l_rate = 1e-3

running_train_loss = None
running_rank = ep_voca_size / 2.0

for i, minibatch in enumerate(
        seqtools.prefetch(train_minibatches, 2, max_buffered=20)):
    if i % 10 == 0:
        ep_train_subset, ep_test_subset, _ = sample_episode(
            labels_train, recordings_train, vocabulary, ep_voca_size, shots, batch_size)
        mean_rank = np.mean(evaluate_matching(
            predict_fn([[f[i] for i in ep_train_subset]
                        for f in feat_seqs_train],
                       durations_train[ep_train_subset]),
            predict_fn([[f[i] for i in ep_test_subset]
                        for f in feat_seqs_train],
                       durations_train[ep_test_subset]),
            labels_train[ep_train_subset],
            labels_train[ep_test_subset],
            ep_voca_size, shots))
        running_rank = 0.9 * running_rank + 0.1 * mean_rank
        print("\rloss: {:>2.3f}, rank: {:>2.3f}".format(
                  running_train_loss or 0., running_rank),
              end='', flush=True)

    if i % 100 == 0:
        report[str(i)] = {
            'running_train_loss': running_train_loss,
            'params': lasagne.layers.get_all_param_values(l_linout)
        }
        print('')

    if i % 5000 == 0:
        l_rate *= 0.3

    batch_loss = float(update_fn(*minibatch, l_rate))
    if np.isnan(batch_loss):
        raise ValueError()
    running_train_loss = .99 * running_train_loss + .01 * batch_loss \
        if running_train_loss else batch_loss
