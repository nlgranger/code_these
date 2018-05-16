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
from experiments.siamese_triplet.common import \
    sample_pairs, sample_episode, sample_triplets, \
    contrastive_loss, kernel_loss, triplet_loss, \
    pairs2minibatches, episode2minibatch, triplets2minibatches, \
    evaluate_matching


np.set_printoptions(precision=2)

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

report = shelve.open(os.path.join(cachedir, experiment_name))
report.clear()

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


# Data ----------------------------------------------------------------------------------

feat_seqs_train = [seqtools.gather(skel_feat_seqs, train_subset)]
labels_train = labels[train_subset].astype(np.int32)
recordings_train = recordings[train_subset]
durations_train = durations[train_subset].astype(np.int32)
vocabulary = np.unique(labels_train)

del recordings, labels, durations, skel_feat_seqs


# Model ---------------------------------------------------------------------------------

model_dict = skel_rnn(
    *tuple(f[0][0].shape for f in feat_seqs_train),
    batch_size=batch_size, max_time=max_time,
    encoder_kwargs=encoder_kwargs)

l_linout = model_dict['l_linout']
l_in = model_dict['l_in']
l_duration = model_dict['l_duration']


# Training routines ---------------------------------------------------------------------

if args.loss == "contrastive":
    l_rate_var = T.scalar('l_rate')
    targets_var = T.ivector('targets')
    linout = lasagne.layers.get_output(l_linout, deterministic=False)
    out = T.tanh(linout)

    train_loss = contrastive_loss(out, targets_var, margin=2.).mean()

    params = lasagne.layers.get_all_params(l_linout, trainable=True)
    updates = lasagne.updates.adam(
        train_loss, params, learning_rate=l_rate_var)

    update_fn = theano.function(
        [l.input_var for l in l_in] + [targets_var, l_duration.input_var, l_rate_var],
        outputs=train_loss, updates=updates)

    # Minibatches
    pairs, labels = sample_pairs(
        labels_train, recordings_train, vocabulary, 300000, 0.1)

    train_minibatches = pairs2minibatches(feat_seqs_train, durations_train,
                                          pairs, labels, batch_size, max_time)

elif args.loss == "triplet":
    # update function
    l_rate_var = T.scalar('l_rate')
    linout = lasagne.layers.get_output(l_linout, deterministic=False)
    out = T.tanh(linout)

    train_loss = triplet_loss(out[:batch_size - batch_size % 3],
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
    out = T.tanh(linout)

    train_loss = kernel_loss(out, targets_var, ep_voca_size, shots).mean()

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

if args.loss == "pei":
    l_rate_var = T.scalar('l_rate')
    targets_var = T.vector('targets')
    linout = lasagne.layers.get_output(l_linout, deterministic=False)
    out = T.nnet.relu(linout)
    out_size = l_linout.num_units
    x1 = out[0::2]
    x2 = out[1::2]

    v = theano.shared(np.ones((out_size,), dtype=np.float32) / np.sqrt(out_size))
    c = theano.shared(np.float32(0))
    similarity = lasagne.nonlinearities.sigmoid(
        T.sum(x1 * x2 * v[None, :], axis=1) + c)
    train_loss = lasagne.objectives.binary_crossentropy(similarity, targets_var).mean()

    params = lasagne.layers.get_all_params(l_linout, trainable=True) + [v, c]
    updates = lasagne.updates.adam(
        train_loss, params, learning_rate=l_rate_var)

    update_fn = theano.function(
        [l.input_var for l in l_in] + [targets_var, l_duration.input_var, l_rate_var],
        outputs=train_loss, updates=updates)

    # Minibatches
    pairs, labels = sample_pairs(
        labels_train, recordings_train, vocabulary, 300000, 0.1)
    labels = labels.astype(np.float32)

    train_minibatches = pairs2minibatches(feat_seqs_train, durations_train,
                                          pairs, labels, batch_size, max_time)

else:
    raise ValueError("unsupported loss argument")

predict_fn = build_predict_fn(model_dict, batch_size, max_time)


# Training iterations -------------------------------------------------------------------
l_rate = 1e-3

running_train_loss = None
running_rank = ep_voca_size / 2.0

for i, minibatch in enumerate(
        seqtools.prefetch(train_minibatches, 2, max_buffered=20)):

    if i % 1 == 0:
        ep_train_subset, ep_test_subset, _ = sample_episode(
            labels_train, recordings_train, vocabulary, ep_voca_size, shots, batch_size)
        z_train = predict_fn([[f[i] for i in ep_train_subset] for f in feat_seqs_train],
                             durations_train[ep_train_subset])
        z_test = predict_fn([[f[i] for i in ep_test_subset] for f in feat_seqs_train],
                            durations_train[ep_test_subset])
        if args.loss == "pei":
            weights = np.sqrt(v.get_value())
            z_train *= weights[None, :]
            z_test *= weights[None, :]
            mean_rank = np.mean(evaluate_matching(
                z_train, z_test,
                labels_train[ep_train_subset], labels_train[ep_test_subset],
                ep_voca_size, shots))
        running_rank = 0.9 * running_rank + 0.1 * mean_rank
        print("\rloss: {:>2.3f}, rank: {:>2.3f}".format(
                  running_train_loss or 0., running_rank),
              end='', flush=True)

    if i % 100 == 0:
        epoch_report = {
            'running_train_loss': running_train_loss,
            'params': lasagne.layers.get_all_param_values(l_linout)}
        if args.loss == 'pei':
            epoch_report['metric_params'] = (v.get_value(), c.get_value())

        report[str(i)] = epoch_report
        print('')

    if (i + 1) % 5000 == 0:
        l_rate *= 0.3

    batch_loss = float(update_fn(*minibatch, l_rate))
    if np.isnan(batch_loss):
        raise ValueError()
    running_train_loss = .99 * running_train_loss + .01 * batch_loss \
        if running_train_loss else batch_loss
