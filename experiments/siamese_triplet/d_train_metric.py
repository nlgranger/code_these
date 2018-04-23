import os
import shelve
import numpy as np
import theano
import theano.tensor as T
import lasagne
import seqtools

from sltools.nn_utils import cdist, adjust_length


def sample_episode(labels, recordings, vocabulary, voca_size, shots, test_size):
    """Sample a training episode.

    :param labels:
        labels from the training set
    :param recordings:
        indexes shared by all variations of an augmented sequence
    :param vocabulary:
        unique labels
    :param voca_size:
        number of classes in one episode (must be smaller than vocabulary)
    :param shots:
        number of training samples for each classes in one episode.
    :param test_size:
        number of test samples in the episode
    :return:
        A tuple containing:

        - voca_size * shots indexes of training samples, grouped by labels
        - test_size indexes of testing samples
        - their labels (within the episode, therefore in [0, voca_size - 1])
    """
    ep_vocabulary = np.random.choice(vocabulary, size=voca_size, replace=False)

    ep_train_subset = []
    ep_test_subset = []
    for l in ep_vocabulary:
        where_label = np.random.permutation(np.where(labels == l)[0])
        non_redundants = np.invert(np.isin(recordings[where_label[shots:]],
                                           recordings[where_label[:shots]]))
        ep_train_subset.extend(where_label[:shots])
        ep_test_subset.extend(where_label[shots:][non_redundants])

    ep_train_subset = np.array(ep_train_subset)
    ep_test_subset = np.random.choice(ep_test_subset, size=test_size, replace=False)
    ep_test_labels = np.argmax(labels[ep_test_subset, None] == ep_vocabulary, axis=1)
    ep_test_labels = ep_test_labels.astype(np.int32)

    return ep_train_subset, ep_test_subset, ep_test_labels


def make_minibatches(feat_seqs, durations, labels, recordings,
                     batch_size, max_time, ep_voca_size, shots, n):
    vocabulary = np.unique(labels)
    n_modalities = len(feat_seqs)

    feat_seqs = [seqtools.smap(lambda s: adjust_length(s, max_time), f)
                 for f in feat_seqs]
    durations = np.fmin(durations, max_time)

    episodes = [sample_episode(labels, recordings, vocabulary, ep_voca_size, shots,
                               batch_size - shots * ep_voca_size)
                for _ in range(n)]

    columns = []

    for m in range(n_modalities):  # modalities of the input
        columns.append(seqtools.starmap(
            lambda ep_idx_train, ep_idx_test, ep_labels:
                np.stack([feat_seqs[m][i] for i in ep_idx_train]
                         + [feat_seqs[m][i] for i in ep_idx_test],
                         axis=0),
            episodes))

    columns.append(seqtools.starmap(
        lambda ep_idx_train, ep_idx_test, ep_labels:
            np.concatenate([durations[ep_idx_train], durations[ep_idx_test]]),
        episodes))

    columns.append(seqtools.starmap(
        lambda ep_idx_train, ep_idx_test, ep_labels:
            ep_labels,
        episodes))

    return seqtools.collate(columns)


def main():
    from experiments.siamese_triplet.a_data import cachedir, durations, labels, \
        recordings, train_subset
    from experiments.siamese_triplet.b_preprocess import skel_feat_seqs
    from experiments.siamese_triplet.c_model import skel_rnn

    report = shelve.open(os.path.join(cachedir, "rnn_report"))
    report.clear()

    # Data
    feat_seqs_train = [seqtools.gather(skel_feat_seqs, train_subset)]
    labels_train = labels[train_subset].astype(np.int32)
    recordings_train = recordings[train_subset]
    durations_train = durations[train_subset].astype(np.int32)

    # feat_seqs_val = [
    #     seqtools.gather(skel_feat_seqs, val_subset)
    # ]
    # labels_val = labels[val_subset].astype(np.int32)
    # durations_val = durations[val_subset].astype(np.int32)
    #
    # vocabulary = np.unique(labels)

    del recordings, labels, durations, skel_feat_seqs

    # Model

    max_time = 128
    batch_size = 64
    shots = 2
    ep_voca_size = 20
    encoder_kwargs = {
        "tconv_sz": 7,
        "filter_dilation": 2,
        "num_tc_filters": 256,
        'dropout': 0.3
    }

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
    l_rate_var = T.scalar('l_rate')
    targets_var = T.ivector('labels')
    linout = lasagne.layers.get_output(l_linout, deterministic=False)

    ep_train_embeddings = linout[:ep_voca_size * shots]
    ep_test_embeddings = linout[ep_voca_size * shots:]

    alpha = 1 - cdist(ep_test_embeddings, ep_train_embeddings, metric='cosine')
    alpha /= alpha.sum(axis=1, keepdims=True)
    alpha = T.reshape(alpha, (ep_test_embeddings.shape[0], ep_voca_size, shots))
    alpha = T.sum(alpha, axis=2)  # average over shots

    train_loss = lasagne.objectives.categorical_crossentropy(alpha, targets_var).sum()

    params = lasagne.layers.get_all_params(l_linout, trainable=True)
    updates = lasagne.updates.adam(train_loss, params, learning_rate=l_rate_var)
    update_fn = theano.function(
        [l.input_var for l in l_in] + [l_duration.input_var, targets_var, l_rate_var],
        outputs=train_loss, updates=updates)

    # linout = lasagne.layers.get_output(l_linout, deterministic=True)
    #
    # ep_train_embeddings = T.reshape(
    #     linout[:ep_voca_size * shots],
    #     (ep_voca_size, shots, -1))
    # ep_test_embeddings = linout[ep_voca_size * shots:]
    #
    # alpha = 1 - cdist(ep_test_embeddings, ep_train_embeddings, metric='cosine')
    # alpha /= alpha.sum(axis=1, keepdims=True)
    # alpha = alpha.reshape(ep_test_embeddings.shape[0], ep_voca_size, shots)
    # alpha = T.sum(alpha, axis=2)  # average over shots
    #
    # test_loss = lasagne.objectives.categorical_crossentropy(alpha, targets_var).sum()
    # loss_fn = theano.function(
    #     [l.input_var for l in l_in] + [l_duration.input_var, l_rate_var],
    #     outputs=test_loss)

    # Minibatches
    train_minibatches = make_minibatches(
        feat_seqs_train, durations_train, labels_train, recordings_train,
        batch_size, max_time, ep_voca_size, shots, 20000)

    # Training iterations
    l_rate = 1e-3

    running_train_loss = np.log(ep_voca_size) * shots * ep_voca_size

    for i, minibatch in enumerate(
            seqtools.prefetch(train_minibatches, 2, max_buffered=20)):
        batch_loss = float(update_fn(*minibatch, l_rate))
        if np.isnan(batch_loss):
            raise ValueError()
        running_train_loss = .99 * running_train_loss + .01 * batch_loss

        if (i + 1) % 10 == 0:
            print("loss: {:>2.3f}".format(running_train_loss))

        if (i + 1) % 100 == 0:
            report[str(i)] = {
                'running_train_loss': running_train_loss,
                'params': lasagne.layers.get_all_param_values(l_linout)
            }

        if (i + 1) % 5000 == 0:
            l_rate *= 0.3


if __name__ == "__main__":
    main()
