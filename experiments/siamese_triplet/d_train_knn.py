import os
import shelve
import numpy as np
import theano
import theano.tensor as T
import lasagne
import seqtools

from sltools.models.triplet import triplet_loss, sample_triplets, triplet2minibatches

from experiments.siamese_triplet.a_data import cachedir, durations, labels, recordings, \
    train_subset, val_subset
from experiments.siamese_triplet.b_preprocess import skel_feat_seqs
from experiments.siamese_triplet.c_model import skel_rnn


report = shelve.open(os.path.join(cachedir, "rnn_report"))
report.clear()

# Data
_, unique_indices = np.unique(recordings[train_subset], return_index=True)
feat_seqs_train = [
    seqtools.gather(skel_feat_seqs, train_subset[unique_indices])
]
labels_train = labels[train_subset[unique_indices]].astype(np.int32)
durations_train = durations[train_subset[unique_indices]].astype(np.int32)

feat_seqs_val = [
    seqtools.gather(skel_feat_seqs, val_subset)
]
labels_val = labels[val_subset].astype(np.int32)
durations_val = durations[val_subset].astype(np.int32)

del recordings, labels, durations, skel_feat_seqs

# Model

max_time = 128
batch_size = 12
encoder_kwargs = {
    "tconv_sz": 9,
    "filter_dilation": 2,
    "num_tc_filters": 256,
    "dropout": 0.1
}

assert batch_size % 3 == 0, "the model must take triplets"

model_dict = skel_rnn(
    *tuple(f[0][0].shape for f in feat_seqs_train),
    batch_size=batch_size, max_time=max_time,
    encoder_kwargs=encoder_kwargs)

l_linout = model_dict['l_linout']
l_in = model_dict['l_in']
l_duration = model_dict['l_duration']

report['meta'] = {
    'modality': 'skel',
    'max_time': max_time,
    'batch_size': batch_size,
    'encoder_kwargs': encoder_kwargs
}

# Training routines

l_rate = 1e-3

l_rate_var = T.scalar('l_rate')
linout = lasagne.layers.get_output(l_linout, deterministic=False)
train_loss = triplet_loss(linout[0::3], linout[1::3], linout[2::3], delta=.4).sum()
params = lasagne.layers.get_all_params(l_linout, trainable=True)
updates = lasagne.updates.adam(train_loss, params, learning_rate=l_rate_var)
update_fn = theano.function(
    [l.input_var for l in l_in] + [l_duration.input_var, l_rate_var],
    outputs=train_loss, updates=updates)

linout = lasagne.layers.get_output(l_linout, deterministic=True)
test_loss = triplet_loss(linout[0::3], linout[1::3], linout[2::3], delta=.4).sum()
loss_fn = theano.function(
    [l.input_var for l in l_in] + [l_duration.input_var],
    outputs=test_loss)

# Training epochs

running_train_loss = 2
running_val_loss = 2

for e in range(50):
    train_losses = []
    val_losses = []

    # Minibatch pipeline

    triplets = sample_triplets(
        sorted(set(labels_train)), labels_train, len(labels_train),
        lambda *_: True)
    train_minibatches = triplet2minibatches(
        feat_seqs_train, durations_train, triplets, batch_size, max_time)
    triplets = sample_triplets(
        sorted(set(labels_val)), labels_val, len(labels_val),
        test=lambda *_: True)
    val_minibatches = triplet2minibatches(
        feat_seqs_val, durations_val, triplets, batch_size, max_time)

    # Minibatch iterations

    for i, minibatch in enumerate(
            seqtools.prefetch(train_minibatches, 2, max_buffered=20)):
        batch_loss = float(update_fn(*minibatch, l_rate))
        if np.isnan(batch_loss):
            raise ValueError()
        running_train_loss = .99 * running_train_loss + .01 * batch_loss

        if i % 3 == 0:
            train_losses.append(batch_loss)

        if i % 30 == 0:
            batch_losses = [loss_fn(*val_minibatches[j])
                            for j in np.random.choice(len(val_minibatches), 10)]
            val_losses.extend(batch_losses)
            running_val_loss = .91 * running_val_loss + .09 * np.mean(batch_losses)
            print("loss: {:>2.3f} / {:>2.3f}".format(running_train_loss,
                                                     running_val_loss))

    # Update setup

    l_rate *= 0.3

    # Report

    report[str(e)] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epoch_loss': running_train_loss,
        'params': lasagne.layers.get_all_param_values(l_linout)
    }
