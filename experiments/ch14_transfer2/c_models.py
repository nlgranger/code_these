import os
import shelve
import pickle as pkl
import lasagne
from lproc import SerializableFunc
from sltools.tconv import TemporalConv
# from experiments.ch14_skel.a_data import tmpdir as rnn_tmpdir
# from experiments.ch14_skel.c_models import build_lstm
from experiments.ch14_shorttc.a_data import tmpdir as rnn_tmpdir
from experiments.ch14_shorttc.c_models import build_lstm


def params_from_rnn(*input_shape):
    max_time = 128
    batch_size = 16

    report = shelve.open(os.path.join(rnn_tmpdir, 'rnn_report'))
    best_epoch = sorted([(float(report[str(e)]['val_scores']['jaccard']), int(e))
                         for e in report.keys() if
                         'val_scores' in report[str(e)].keys()])[-1][1]

    model = build_lstm(*input_shape,
                       batch_size=batch_size, max_time=max_time)

    all_layers = lasagne.layers.get_all_layers(model['l_linout'])
    if 'params' in report[str(best_epoch)].keys():
        params = report[str(best_epoch)]['params']
    else:
        with open(os.path.join(rnn_tmpdir, "rnn_it{:04d}.pkl".format(best_epoch)), 'rb') as f:
            params = pkl.load(f)

    lasagne.layers.set_all_param_values(all_layers, params)

    return lasagne.layers.get_all_param_values(
        all_layers[all_layers.index(model['l_in']) + 1
                   :all_layers.index(model['l_feats'])])


@SerializableFunc
def build_encoder(l_in, params=None, freeze=False):
    dropout = 0.3
    tconv_sz = 3
    filter_dilation = 1
    warmup = (tconv_sz * filter_dilation) // 2

    l1 = lasagne.layers.DenseLayer(
        l_in, num_units=1024,
        num_leading_axes=2,
        nonlinearity=lasagne.nonlinearities.leaky_rectify)
    l1 = lasagne.layers.batch_norm(l1, axes=(0, 1))
    l1 = lasagne.layers.dropout(l1, p=dropout)

    l2 = lasagne.layers.DenseLayer(
        l1, num_units=1024,
        num_leading_axes=2,
        nonlinearity=lasagne.nonlinearities.leaky_rectify)
    l2 = lasagne.layers.batch_norm(l2, axes=(0, 1))
    l2 = lasagne.layers.dropout(l2, p=dropout)

    l3 = TemporalConv(l2, num_filters=256, filter_size=tconv_sz,
                      filter_dilation=filter_dilation, pad='same',
                      conv_type='regular',
                      nonlinearity=lasagne.nonlinearities.leaky_rectify)
    l3 = lasagne.layers.batch_norm(l3, axes=(0, 1))

    if params is not None:
        layers = lasagne.layers.get_all_layers(l3)
        layers = layers[layers.index(l_in) + 1:]
        lasagne.layers.set_all_param_values(layers, params)

    if freeze:
        layers = lasagne.layers.get_all_layers(l3)
        layers = layers[layers.index(l_in) + 1:]

        for l in layers:
            for param in l.params:
                l.params[param].discard('trainable')

    return {
        'l_out': l3,
        'warmup': warmup
    }
