import numpy as np
import lasagne
from lasagne.layers import Conv2DLayer, NonlinearityLayer, \
    ElemwiseSumLayer, GlobalPoolLayer, ReshapeLayer, BatchNormLayer, DropoutLayer
from lasagne.nonlinearities import rectify, leaky_rectify
import theano.tensor as T
from lproc import SerializableFunc

from sltools.nn_utils import DurationMaskLayer
from sltools.tconv import TemporalConv
from sltools.modrop import modrop


# Model parts ---------------------------------------------------------------------------

def wide_resnet(l_in, d, k, dropout=0.):
    """Build a Wide-Resnet WRN-d-k

    Parameters
    ----------
    :param l_in:
        input Layer
    :param d:
        network depth (d follow the relation d = 6 * n + 4 where n is the number of blocs
        by groups)
    :param k:
        widening factor
    :param dropout:
        dropout rate
    """
    if (d - 4) % 6 != 0:
        raise ValueError("d should be of the form d = 6 * n + 4")

    n = (d - 4) // 6
    he_norm = lasagne.init.HeNormal(gain='relu')

    def basic_block(incoming, num_filters, stride, shortcut, name=None):
        name = name + "_" if name is not None else ""

        conv_path = BatchNormLayer(incoming)
        conv_path = NonlinearityLayer(conv_path, nonlinearity=rectify)

        rectified_input = conv_path  # reused in linear shortcut

        # TODO: not clear if we should dropout here, authors code doesn't seem to

        conv_path = Conv2DLayer(
            conv_path, num_filters=num_filters, filter_size=(3, 3),
            stride=stride, pad='same',
            W=he_norm, b=None, nonlinearity=None, name=name + "conv1")

        conv_path = BatchNormLayer(conv_path)
        conv_path = NonlinearityLayer(conv_path, nonlinearity=rectify)
        if dropout > 0:
            conv_path = DropoutLayer(conv_path, p=dropout)

        conv_path = Conv2DLayer(
            conv_path, num_filters=num_filters, filter_size=(3, 3),
            pad='same',
            W=he_norm, b=None, nonlinearity=None, name=name + "conv2")

        if shortcut == 'identity':
            assert stride == (1, 1) or stride == 1
            short_path = incoming

        elif shortcut == 'linear':
            short_path = Conv2DLayer(
                rectified_input, num_filters=num_filters, filter_size=(1, 1),
                stride=stride, pad='same',
                W=he_norm, b=None, nonlinearity=None)

        else:
            raise ValueError("invalid parameter value for shortcut")

        o = ElemwiseSumLayer([conv_path, short_path], name=name + "sum")
        return o

    net = Conv2DLayer(
        l_in, num_filters=16, filter_size=(3, 3),
        pad='same',
        W=he_norm, b=None, nonlinearity=None)

    net = basic_block(net, 16 * k, stride=(1, 1), shortcut='linear',
                      name="block11")
    for i in range(1, n):
        net = basic_block(net, 16 * k, stride=(1, 1), shortcut='identity',
                          name="block1" + str(i + 1))

    net = basic_block(net, 32 * k, stride=(2, 2), shortcut='linear',
                      name="block21")
    for i in range(1, n):
        net = basic_block(net, 32 * k, stride=(1, 1), shortcut='identity',
                          name="block2" + str(i + 1))

    net = basic_block(net, 64 * k, stride=(2, 2), shortcut='linear',
                      name="block31")
    for i in range(1, n):
        net = basic_block(net, 64 * k, stride=(1, 1), shortcut='identity',
                          name="block3" + str(i + 1))

    net = BatchNormLayer(net)
    net = NonlinearityLayer(net, nonlinearity=rectify)

    net = GlobalPoolLayer(net, T.mean, name="MeanPool")

    return net


@SerializableFunc
def skel_encoder(l_in, dropout=0.3, tconv_sz=17, filter_dilation=1):
    warmup = (tconv_sz * filter_dilation) // 2

    l1 = lasagne.layers.DenseLayer(
        l_in, num_units=1024,
        num_leading_axes=2,
        nonlinearity=None)
    l1 = BatchNormLayer(l1, axes=(0, 1))
    l1 = NonlinearityLayer(l1, leaky_rectify)

    d1 = DropoutLayer(l1, p=dropout)

    l2 = lasagne.layers.DenseLayer(
        d1, num_units=1024,
        num_leading_axes=2,
        nonlinearity=None)
    l2 = BatchNormLayer(l2, axes=(0, 1))
    l2 = NonlinearityLayer(l2, leaky_rectify)

    d2 = DropoutLayer(l2, p=dropout)

    l3 = TemporalConv(d2, num_filters=172, filter_size=tconv_sz,
                      filter_dilation=filter_dilation, pad='same',
                      conv_type='regular',
                      nonlinearity=None)
    l3 = BatchNormLayer(l3, axes=(0, 1))
    l3 = NonlinearityLayer(l3, leaky_rectify)

    return {
        'l_out': l3,
        'warmup': warmup
    }


@SerializableFunc
def bgr_encoder(l_in, dropout=0., tconv_sz=17, filter_dilation=1):
    warmup = (tconv_sz * filter_dilation) // 2
    batch_size, max_time, _, *crop_size = l_in.output_shape

    # stack pairs of small images into one batch of images
    l_r1 = ReshapeLayer(l_in, (-1, 1) + crop_size)

    # process through (siamese) CNN
    l_cnnout = wide_resnet(l_r1, d=16, k=1, dropout=dropout)

    # Concatenate feature vectors from the pairs
    feat_shape = np.asscalar(np.prod(l_cnnout.output_shape[1:]))
    l_feats = ReshapeLayer(l_cnnout, (batch_size, max_time, 2 * feat_shape))

    l_drop1 = DropoutLayer(l_feats, p=dropout)

    l_out = TemporalConv(l_drop1, num_filters=2 * feat_shape, filter_size=tconv_sz,
                         filter_dilation=filter_dilation, pad='same',
                         conv_type='match',
                         nonlinearity=None)
    l_out = BatchNormLayer(l_out, axes=(0, 1))
    l_out = NonlinearityLayer(l_out, leaky_rectify)

    return {
        'l_out': l_out,
        'warmup': warmup
    }


@SerializableFunc
def fusion_encoder(l_in_skel, l_in_zmaps, **kwargs):
    modropout = [0.3, 0.1]

    skel_feats = skel_encoder(l_in_skel, **kwargs)
    zmaps_feats = bgr_encoder(l_in_zmaps, **kwargs)

    l_skel_feats, l_zmaps_feats = \
        modrop([skel_feats['l_out'], zmaps_feats['l_out']], p=modropout)

    l_feats = lasagne.layers.ConcatLayer([l_skel_feats, l_zmaps_feats], axis=2)

    return {
        'l_out': l_feats,
        'warmup': max(skel_feats['warmup'], zmaps_feats['warmup'])
    }


@SerializableFunc
def identity_encoder(l_in, warmup):
    return {
        'l_out': l_in,
        'warmup': warmup,
    }


# ---------------------------------------------------------------------------------------

@SerializableFunc
def skel_lstm(feats_shape, batch_size=6, max_time=64, encoder_kwargs=None):
    encoder_kwargs = encoder_kwargs or {}
    n_lstm_units = 172

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, max_time) + feats_shape, name="l_in")
    encoder_data = skel_encoder(l_in, **encoder_kwargs)
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

    l_d1 = lasagne.layers.dropout(l_feats, p=.3)
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
        'l_linout': l_linout,
        # 'l_out': l_lstm,
        'warmup': warmup,
        'l_feats': l_feats
    }


@SerializableFunc
def bgr_lstm(feats_shape, batch_size=6, max_time=64, encoder_kwargs=None):
    encoder_kwargs = encoder_kwargs or {}
    n_lstm_units = 172

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, max_time) + feats_shape, name="l_in")
    encoder_data = bgr_encoder(l_in, **encoder_kwargs)
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

    l_d1 = DropoutLayer(l_feats, p=0.3)
    l_lstm1 = lasagne.layers.GRULayer(
        l_d1, num_units=n_lstm_units, mask_input=l_mask,
        learn_init=True, name='lstm_fw')
    l_lstm2 = lasagne.layers.GRULayer(
        l_d1, num_units=n_lstm_units, mask_input=l_mask,
        backwards=True, learn_init=True, name="lstm_bw")
    l_cc1 = lasagne.layers.ConcatLayer((l_lstm1, l_lstm2), axis=2)
    l_cc1 = DropoutLayer(l_cc1, p=.3)

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
def fusion_lstm(skel_feats_shape, bgr_feats_shape, max_time=64, batch_size=6,
                encoder_kwargs=None):
    encoder_kwargs = encoder_kwargs or {}
    n_lstm_units = 172  # TODO: check if 128 reduces overfitting

    l_in_skel = lasagne.layers.InputLayer(
        shape=(batch_size, max_time) + skel_feats_shape)
    l_in_bgr = lasagne.layers.InputLayer(
        shape=(batch_size, max_time) + bgr_feats_shape)
    encoder_data = fusion_encoder(l_in_skel, l_in_bgr, **encoder_kwargs)
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

    l_d1 = DropoutLayer(l_feats, p=0.3)
    l_lstm1 = lasagne.layers.GRULayer(
        l_d1, num_units=n_lstm_units, mask_input=l_mask,
        grad_clipping=1., learn_init=True)
    l_lstm2 = lasagne.layers.GRULayer(
        l_d1, num_units=n_lstm_units, mask_input=l_mask,
        backwards=True, grad_clipping=1., learn_init=True)
    l_cc1 = lasagne.layers.ConcatLayer((l_lstm1, l_lstm2), axis=2)
    l_cc1 = DropoutLayer(l_cc1, p=.3)

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
def transfer_lstm(feats_shape, batch_size=6, max_time=64, warmup=0):
    n_lstm_units = 172

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, max_time) + feats_shape, name="l_in")
    encoder_data = identity_encoder(l_in, warmup)
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

    l_d1 = lasagne.layers.dropout(l_feats, p=.3)
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
        'l_linout': l_linout,
        # 'l_out': l_lstm,
        'warmup': warmup,
        'l_feats': l_feats
    }