import os
import shelve

import lasagne
import theano.tensor as T
from lproc import SerializableFunc
from experiments.ch14_bgr.a_data import tmpdir as bgr_tmpdir
from experiments.ch14_bgr.c_models import build_encoder as build_zmaps_encoder
from experiments.ch14_skel.a_data import tmpdir as skel_tmpdir

from experiments.ch14_skel.c_models import build_encoder as build_skel_encoder
from sltools.modrop import modrop
from sltools.nn_utils import DurationMaskLayer


@SerializableFunc
def build_encoder(l_in_skel, l_in_zmaps,
                  skel_params=None, bgr_params=None, freeze=False):
    modropout = [0.3, 0.1]
    param_source = None
    skel_iter = 119
    bgr_iter = 114

    if param_source == "rnn":
        report = shelve.open(os.path.join(skel_tmpdir, "rnn_report"))
        model = report[str(skel_iter)]['model']
        l_feats = model.build_extra['l_feats']
        layers = lasagne.layers.get_all_layers(l_feats)
        layers = layers[layers.index(model.l_in[0]) + 1:]
        skel_params = [lasagne.layers.get_all_param_values(layers)]
        report = shelve.open(os.path.join(bgr_tmpdir, "rnn_report"))
        model = report[str(bgr_iter)]['model']
        l_feats = model.build_extra['l_feats']
        layers = lasagne.layers.get_all_layers(l_feats)
        layers = layers[layers.index(model.l_in[0]) + 1:]
        bgr_params = [lasagne.layers.get_all_param_values(layers)]

    elif param_source == "hmm":
        report = shelve.open(os.path.join(skel_tmpdir, "hmm_report"))
        model = report[str(skel_iter)]['model'].posterior
        l_feats = model.l_feats
        layers = lasagne.layers.get_all_layers(l_feats)
        layers = layers[layers.index(model.l_in[0]) + 1:]
        skel_params = [lasagne.layers.get_all_param_values(layers)]
        report = shelve.open(os.path.join(bgr_tmpdir, "hmm_report"))
        model = report[str(bgr_iter)]['model']
        l_feats = model.build_extra['l_feats']
        layers = lasagne.layers.get_all_layers(l_feats)
        layers = layers[layers.index(model.l_in[0]) + 1:]
        bgr_params = [lasagne.layers.get_all_param_values(layers)]

    skel_feats = build_skel_encoder(l_in_skel, params=skel_params)
    zmaps_feats = build_zmaps_encoder(l_in_zmaps, params=bgr_params)

    l_skel_feats, l_zmaps_feats = \
        modrop([skel_feats['l_out'], zmaps_feats['l_out']], p=modropout)

    l_feats = lasagne.layers.ConcatLayer(
        [l_skel_feats, l_zmaps_feats], axis=2)

    if freeze:
        for layer in lasagne.layers.get_all_layers(l_feats):
            for param in layer.params:
                layer.params[param].discard('trainable')

    return {
        'l_out': l_feats,
        'warmup': max(skel_feats['warmup'], zmaps_feats['warmup'])
    }


@SerializableFunc
def build_lstm(skel_feats_shape, bgr_feats_shape, max_time=64, batch_size=6):
    n_lstm_units = 172

    l_in_skel = lasagne.layers.InputLayer(
        shape=(batch_size, max_time) + skel_feats_shape)
    l_in_bgr = lasagne.layers.InputLayer(
        shape=(batch_size, max_time) + bgr_feats_shape)
    encoder_data = build_encoder(l_in_skel, l_in_bgr)
    l_feats = encoder_data['l_out']
    warmup = encoder_data['warmup']

    # LSTM layers
    durations = T.ivector()
    l_duration = lasagne.layers.InputLayer(
        (batch_size,), input_var=durations,
        name="l_duration")
    l_mask = DurationMaskLayer(
        l_duration, max_time,
        name="l_mask")

    l_d1 = lasagne.layers.dropout(l_feats, p=0.3)
    l_lstm1 = lasagne.layers.GRULayer(
        l_d1, num_units=n_lstm_units, mask_input=l_mask,
        grad_clipping=1., learn_init=True)
    l_lstm2 = lasagne.layers.GRULayer(
        l_feats, num_units=n_lstm_units, mask_input=l_mask,
        backwards=True, grad_clipping=1., learn_init=True)
    l_cc1 = lasagne.layers.ConcatLayer((l_lstm1, l_lstm2), axis=2)
    l_cc1 = lasagne.layers.dropout(l_cc1, p=.3)

    l_linout = lasagne.layers.DenseLayer(
        l_cc1, num_units=21, num_leading_axes=2, nonlinearity=None,
        name="l_linout")

    return {
        'l_in': [l_in_skel, l_in_bgr],
        'l_mask': l_mask,
        'l_duration': l_duration,
        'l_linout': l_linout,
        # 'l_out': l_lstm,
        'warmup': warmup,
        'l_feats': l_feats
    }
