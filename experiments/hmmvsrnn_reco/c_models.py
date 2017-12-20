import numpy as np
import lasagne
from lasagne.layers import Conv2DLayer, NonlinearityLayer, \
    ElemwiseSumLayer, GlobalPoolLayer, SliceLayer, \
    PadLayer, ReshapeLayer, BatchNormLayer, batch_norm
from lasagne.nonlinearities import rectify, leaky_rectify
import theano.tensor as T
from lproc import SerializableFunc

from sltools.nn_utils import DurationMaskLayer
from sltools.tconv import TemporalConv
from sltools.modrop import modrop


# Model parts ---------------------------------------------------------------------------


def resnet_block(incoming, num_filters, is_first_block,
                 stride=(1, 1), shortcut='identity'):
    he_norm = lasagne.init.HeNormal(gain='relu')

    if is_first_block:
        conv_path = incoming
    else:
        conv_path = BatchNormLayer(incoming)
        conv_path = NonlinearityLayer(conv_path, rectify)

    conv_path = Conv2DLayer(
        conv_path, num_filters=num_filters, filter_size=(3, 3),
        stride=stride, pad='same',
        W=he_norm, nonlinearity=None)
    conv_path = BatchNormLayer(conv_path)
    conv_path = NonlinearityLayer(conv_path, nonlinearity=rectify)

    conv_path = Conv2DLayer(
        conv_path, num_filters=num_filters, filter_size=(3, 3),
        pad='same',
        W=he_norm, nonlinearity=None)

    if shortcut == 'identity' and (stride == (1, 1) or stride == 1):
        short_path = incoming

    elif shortcut == 'identity':
        short_path = SliceLayer(
            incoming, indices=slice(0, incoming.output_shape[2], 2), axis=2)
        short_path = SliceLayer(
            short_path, indices=slice(0, incoming.output_shape[3], 2), axis=3)
        short_path = PadLayer(
            short_path, batch_ndim=1,
            width=[(0, num_filters - incoming.output_shape[1])])

    elif shortcut == 'linear':
        short_path = Conv2DLayer(
            incoming, num_filters=num_filters, filter_size=(1, 1),
            stride=stride, pad='same',
            W=he_norm, nonlinearity=None)

    else:
        raise ValueError("invalid parameter value for shortcut")

    assert all(s1 == s2 for s1, s2
               in zip(conv_path.output_shape, short_path.output_shape))

    out = ElemwiseSumLayer([conv_path, short_path])

    if not is_first_block:
        out = BatchNormLayer(out)
        out = NonlinearityLayer(out, rectify)

    return out


def wide_resnet(l_in, n, k, shortcut="identity"):
    net = Conv2DLayer(
        l_in, num_filters=16, filter_size=(3, 3),
        pad='same', nonlinearity=rectify)
    net = batch_norm(net)

    net = resnet_block(net, 16 * k, True, stride=(2, 2), shortcut=shortcut)
    for i in range(n):
        net = resnet_block(net, 16 * k, False, shortcut=shortcut)

    net = resnet_block(net, 32 * k, True, stride=(2, 2), shortcut=shortcut)
    for i in range(1, n):
        net = resnet_block(net, 32 * k, False, shortcut=shortcut)

    net = resnet_block(net, 64 * k, True, stride=(2, 2))
    for i in range(1, n):
        net = resnet_block(net, 64 * k, False, shortcut=shortcut)

    net = GlobalPoolLayer(net)

    return net


@SerializableFunc
def skel_encoder(l_in):
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

    return {
        'l_out': l3,
        'warmup': warmup
    }


@SerializableFunc
def bgr_encoder(l_in):
    tconv = 17
    warmup = tconv // 2
    filter_dilation = 1

    # stack pairs of small images into one batch of images
    l_r1 = ReshapeLayer(l_in, (-1, 1) + l_in.output_shape[3:])

    # process through (siamese) CNN
    l_cnnout = wide_resnet(l_r1, n=2, k=2)

    # Concatenate feature vectors from the pairs
    feat_shape = np.asscalar(np.prod(l_cnnout.output_shape[1:]))
    l_feats = ReshapeLayer(
        l_cnnout, (l_in.output_shape[0], l_in.output_shape[1], 2 * feat_shape))
    l_feats = NonlinearityLayer(l_feats, leaky_rectify)
    l_feats = batch_norm(l_feats, axes=(0, 1))

    l_out = TemporalConv(l_feats, num_filters=l_feats.output_shape[-1],
                         filter_size=tconv, filter_dilation=filter_dilation,
                         pad='same', conv_type='match',
                         nonlinearity=leaky_rectify)
    l_out = batch_norm(l_out, axes=(0, 1))

    return {
        'l_out': l_out,
        'warmup': warmup
    }


@SerializableFunc
def fusion_encoder(l_in_skel, l_in_zmaps):
    modropout = [0.3, 0.1]

    skel_feats = skel_encoder(l_in_skel)
    zmaps_feats = bgr_encoder(l_in_zmaps)

    l_skel_feats, l_zmaps_feats = \
        modrop([skel_feats['l_out'], zmaps_feats['l_out']], p=modropout)

    l_feats = lasagne.layers.ConcatLayer([l_skel_feats, l_zmaps_feats], axis=2)

    return {
        'l_out': l_feats,
        'warmup': max(skel_feats['warmup'], zmaps_feats['warmup'])
    }


# ---------------------------------------------------------------------------------------

@SerializableFunc
def skel_lstm(feats_shape, batch_size=6, max_time=64):
    n_lstm_units = 172

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, max_time) + feats_shape, name="l_in")
    encoder_data = skel_encoder(l_in)
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
        'l_in': [l_in],
        'l_duration': l_duration,
        'l_mask': l_mask,
        'l_linout': l_linout,
        # 'l_out': l_lstm,
        'warmup': warmup,
        'l_feats': l_feats
    }


@SerializableFunc
def build_lstm(feats_shape, batch_size=6, max_time=64):
    n_lstm_units = 172

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, max_time) + feats_shape, name="l_in")
    encoder_data = bgr_encoder(l_in)
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
        'l_in': [l_in],
        'l_duration': l_duration,
        'l_mask': l_mask,
        'l_linout': l_linout,
        # 'l_out': l_lstm,
        'warmup': warmup,
        'l_feats': l_feats
    }


@SerializableFunc
def fusion_lstm(skel_feats_shape, bgr_feats_shape, max_time=64, batch_size=6):
    n_lstm_units = 172

    l_in_skel = lasagne.layers.InputLayer(
        shape=(batch_size, max_time) + skel_feats_shape)
    l_in_bgr = lasagne.layers.InputLayer(
        shape=(batch_size, max_time) + bgr_feats_shape)
    encoder_data = fusion_encoder(l_in_skel, l_in_bgr)
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


@SerializableFunc
def transfer_skel_lstm(feats_shape, hmm_report, batch_size=6, max_time=64):
    phases = list(hmm_report.keys())
    best_phase = phases[int(np.argmax([hmm_report[p]['val_report']['jaccard']
                                       for p in phases]))]
    hmm_recognizer = hmm_report[best_phase]['model']
    n_lstm_units = 172

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, max_time) + feats_shape, name="l_in")
    encoder_data = skel_encoder(l_in)
    l_feats = encoder_data['l_out']
    warmup = encoder_data['warmup']

    # transfer (posteriors)
    # l_feats = lasagne.layers.DenseLayer(
    #     l_feats, hmm_recognizer.nstates, num_leading_axes=2,
    #     nonlinearity=lambda x: T.exp(log_softmax(x)))
    #
    # src_layers = lasagne.layers.get_all_layers(hmm_recognizer.posterior.l_out)
    # src_layers = src_layers[src_layers.index(hmm_recognizer.posterior.l_in[-1]) + 1:]
    # src_params = lasagne.layers.get_all_param_values(src_layers)
    # tgt_layers = lasagne.layers.get_all_layers(l_feats)
    # tgt_layers = tgt_layers[tgt_layers.index(l_in) + 1:]
    # lasagne.layers.set_all_param_values(tgt_layers, src_params)

    # transfer (features)
    src_layers = lasagne.layers.get_all_layers(hmm_recognizer.posterior.l_feats)
    src_layers = src_layers[src_layers.index(hmm_recognizer.posterior.l_in[-1]) + 1:]
    src_params = lasagne.layers.get_all_param_values(src_layers)
    tgt_layers = lasagne.layers.get_all_layers(l_feats)
    tgt_layers = tgt_layers[tgt_layers.index(l_in) + 1:]
    lasagne.layers.set_all_param_values(tgt_layers, src_params)

    # freeze
    for l in tgt_layers:
        for p in l.params:
            l.params[p].discard('trainable')

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
        l_d1, num_units=n_lstm_units, mask_input=l_mask,
        backwards=True, grad_clipping=1., learn_init=True)
    l_cc1 = lasagne.layers.ConcatLayer((l_lstm1, l_lstm2), axis=2)
    l_cc1 = lasagne.layers.dropout(l_cc1, p=.3)

    l_linout = lasagne.layers.DenseLayer(
        l_cc1, num_units=21, num_leading_axes=2, nonlinearity=None,
        name="l_linout")

    return {
        'l_in': [l_in],
        'l_duration': l_duration,
        'l_mask': l_mask,
        'l_cc': l_cc1,
        'l_linout': l_linout,
        # 'l_out': l_lstm,
        'warmup': warmup,
        'l_feats': l_feats
    }
