import os
import shelve
import pickle as pkl
import numpy as np
import lasagne
import theano.tensor as T
from sltools.nn_utils import DurationMaskLayer
from sltools.tconv import TemporalConv
from experiments.ch14_skel.a_data import tmpdir as skel_tmpdir
from experiments.ch14_skel.c_models import build_lstm as build_skel_lstm


def build_encoder(l_in, params=None, freeze=False):
    dropout = 0.3
    tconv_sz = 17
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


def pretrained_skel_params(input_shape):
    report = shelve.open(os.path.join(skel_tmpdir, 'rnn_report'))
    best_epoch = np.argmax([float(report[str(e)]['val_scores']['jaccard'])
                            for e in report.keys()
                            if 'val_scores' in report[str(e)].keys()])
    print("reloading parameters from RNN at it {}".format(best_epoch))
    model = build_skel_lstm(input_shape, batch_size=1, max_time=1)
    all_layers = lasagne.layers.get_all_layers(model['l_linout'])

    param_dump_file = os.path.join(skel_tmpdir, "rnn_it{:04d}.pkl".format(best_epoch))
    with open(param_dump_file, 'rb') as f:
        params = pkl.load(f)
        lasagne.layers.set_all_param_values(all_layers, params)

    i1 = all_layers.index(model['l_in'])
    i2 = all_layers.index(model['l_cc'])
    return lasagne.layers.get_all_param_values(all_layers[i1+1:i2])


def build_model(feat_shape, batch_size, max_time):
    n_lstm_units = 172

    # Input layers
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, max_time) + feat_shape)

    l_duration = lasagne.layers.InputLayer(
        (batch_size,), input_var=T.ivector())
    l_mask = DurationMaskLayer(l_duration, max_time, name="l_mask")

    # feature encoding/representation learning
    encoder_data = build_encoder(l_in)
    l_feats = encoder_data['l_out']
    l_feats = lasagne.layers.dropout(l_feats, p=0.3)
    warmup = encoder_data['warmup']

    # LSTM layers
    l_lstm1 = lasagne.layers.GRULayer(
        l_feats, num_units=n_lstm_units, mask_input=l_mask,
        grad_clipping=1., learn_init=True, only_return_final=True)
    l_lstm2 = lasagne.layers.GRULayer(
        l_feats, num_units=n_lstm_units, mask_input=l_mask,
        backwards=True, grad_clipping=1., learn_init=True, only_return_final=True)
    l_cc1 = lasagne.layers.ConcatLayer((l_lstm1, l_lstm2), axis=1)
    l_cc1 = lasagne.layers.dropout(l_cc1, p=.3)

    l_f4 = lasagne.layers.DenseLayer(
        l_cc1, num_units=256,
        nonlinearity=lasagne.nonlinearities.sigmoid)
    l_f4 = lasagne.layers.ScaleLayer(
        l_f4,
        scales=np.full((256,), 1/np.sqrt(256), dtype=np.float32))

    # Transfer
    # params = pretrained_skel_params(feat_shape)
    # all_layers = lasagne.layers.get_all_layers(l_cc1)
    # i1 = all_layers.index(l_in)
    # i2 = all_layers.index(l_cc1)
    # lasagne.layers.set_all_param_values(all_layers[i1+1:i2], params)

    return {
        'l_in': l_in,
        'l_duration': l_duration,
        'l_linout': l_f4,
        'warmup': warmup,
        'l_feats': encoder_data['l_out']
    }
