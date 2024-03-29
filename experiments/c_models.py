import os
import shelve
import numpy as np
import lasagne
from lasagne.layers import Conv2DLayer, NonlinearityLayer, \
    ElemwiseSumLayer, GlobalPoolLayer, ReshapeLayer, BatchNormLayer, \
    DropoutLayer, GaussianNoiseLayer
from lasagne.nonlinearities import rectify, leaky_rectify
import theano.tensor as T
from lproc import SerializableFunc

from sltools.nn_utils import DurationMaskLayer, softmax
from sltools.tconv import TemporalConv
from sltools.modrop import modrop


# Model parts ---------------------------------------------------------------------------

def wide_resnet(l_in, d, k):
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

        conv_path = Conv2DLayer(
            conv_path, num_filters=num_filters, filter_size=(3, 3),
            stride=stride, pad='same',
            W=he_norm, b=None, nonlinearity=None, name=name + "conv1")

        conv_path = BatchNormLayer(conv_path)
        conv_path = NonlinearityLayer(conv_path, nonlinearity=rectify)

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
def skel_encoder(l_in, tconv_sz, filter_dilation, num_tc_filters, dropout):
    warmup = 16

    l1 = lasagne.layers.DenseLayer(
        l_in, num_units=480,
        num_leading_axes=2,
        nonlinearity=None, b=None)
    l1 = BatchNormLayer(l1, axes=(0, 1))
    l1 = NonlinearityLayer(l1, leaky_rectify)

    d1 = DropoutLayer(l1, p=dropout)

    l2 = lasagne.layers.DenseLayer(
        d1, num_units=480,
        num_leading_axes=2,
        nonlinearity=None, b=None)
    l2 = BatchNormLayer(l2, axes=(0, 1))
    l2 = NonlinearityLayer(l2, leaky_rectify)

    d2 = DropoutLayer(l2, p=dropout)

    l3 = TemporalConv(d2, num_filters=num_tc_filters, filter_size=tconv_sz,
                      filter_dilation=filter_dilation, pad='same',
                      conv_type='regular',
                      nonlinearity=None, b=None)
    l3 = BatchNormLayer(l3, axes=(0, 1))
    l3 = NonlinearityLayer(l3, leaky_rectify)

    return {
        'l_out': l3,
        'warmup': warmup
    }


@SerializableFunc
def bgr_encoder(l_in, tconv_sz, filter_dilation, num_tc_filters, dropout):
    warmup = 16
    batch_size, max_time, _, *crop_size = l_in.output_shape
    crop_size = tuple(crop_size)

    # stack pairs of small images into one batch of images
    l_r1 = ReshapeLayer(l_in, (-1, 1) + crop_size)

    # process through (siamese) CNN
    l_cnnout = wide_resnet(l_r1, d=16, k=1)

    # Concatenate feature vectors from the pairs
    feat_shape = np.asscalar(np.prod(l_cnnout.output_shape[1:]))
    l_feats = ReshapeLayer(l_cnnout, (batch_size, max_time, 2 * feat_shape))

    if dropout > 0:
        l_feats = DropoutLayer(l_feats, p=dropout)

    l_out = TemporalConv(l_feats, num_filters=num_tc_filters, filter_size=tconv_sz,
                         filter_dilation=filter_dilation, pad='same',
                         b=None, nonlinearity=None)
    l_out = BatchNormLayer(l_out, axes=(0, 1))
    l_out = NonlinearityLayer(l_out, leaky_rectify)

    return {
        'l_out': l_out,
        'warmup': warmup
    }


@SerializableFunc
def fusion_encoder(l_in_skel, l_in_zmaps, skel_kwargs, bgr_kwargs):
    modropout = [0.3, 0.1]

    skel_feats = skel_encoder(l_in_skel, **skel_kwargs)
    zmaps_feats = bgr_encoder(l_in_zmaps, **bgr_kwargs)

    l_skel_feats, l_zmaps_feats = \
        modrop([skel_feats['l_out'], zmaps_feats['l_out']], p=modropout)

    l_feats = lasagne.layers.ConcatLayer([l_skel_feats, l_zmaps_feats], axis=2)

    return {
        'l_out': l_feats,
        'warmup': max(skel_feats['warmup'], zmaps_feats['warmup'])
    }


@SerializableFunc
def transfer_encoder(*l_in, transfer_from, freeze_at, terminate_at):
    # NOTE: this model has its own noise layer at the top !!!
    from experiments.a_data import cachedir
    from experiments.utils import reload_best_hmm, reload_best_rnn

    report = shelve.open(os.path.join(cachedir, transfer_from))

    if report['meta']['model'] == "hmm":
        # reload pretrained model
        _, recognizer, _ = reload_best_hmm(report)
        posterior = recognizer.posterior

        if freeze_at == "embedding":
            # build model
            l_embedding = l_in[0]
            l_embedding = lasagne.layers.DropoutLayer(l_embedding, p=0.3)

            l_logits = lasagne.layers.DenseLayer(
                l_embedding, posterior.nstates,
                num_leading_axes=2, nonlinearity=None)

            l_posteriors = lasagne.layers.NonlinearityLayer(l_logits, softmax)

            # load but don't freeze
            for p1, p2 in zip(posterior.l_raw.get_params(), l_logits.get_params()):
                p2.set_value(p1.get_value())

            if terminate_at == "embedding":
                return {'l_out': l_embedding, 'warmup': posterior.warmup}
            elif terminate_at == "logits":
                return {'l_out': l_logits, 'warmup': posterior.warmup}
            if terminate_at == "posteriors":
                return {'l_out': l_posteriors, 'warmup': posterior.warmup}
            else:
                raise ValueError()

        elif freeze_at == "logits":
            # build model
            l_logits = l_in[0]
            l_logits = GaussianNoiseLayer(l_logits, sigma=4)

            l_posteriors = NonlinearityLayer(l_logits, softmax)

            if terminate_at == "logits":
                return {'l_out': l_logits, 'warmup': posterior.warmup}
            if terminate_at == "posteriors":
                return {'l_out': l_posteriors, 'warmup': posterior.warmup}
            else:
                raise ValueError()

        else:
            raise ValueError

    elif report['meta']['model'] == "rnn":
        # reload pretrained model
        _, recognizer, _ = reload_best_rnn(report)
        old_layers = lasagne.layers.get_all_layers(recognizer["l_feats"])

        if freeze_at == "inputs":
            # build model
            if report['meta']['modality'] == 'skel':
                encoder_dict = skel_encoder(*l_in, **report['args']['encoder_kwargs'])
            elif report['meta']['modality'] == 'bgr':
                encoder_dict = bgr_encoder(*l_in, **report['args']['encoder_kwargs'])
            elif report['meta']['modality'] == 'fusion':
                encoder_dict = fusion_encoder(*l_in, **report['args']['encoder_kwargs'])
            else:
                raise ValueError()

            l_embedding = encoder_dict['l_out']

            # load parameters
            params = lasagne.layers.get_all_param_values(old_layers)
            new_layers = lasagne.layers.get_all_layers(l_embedding)
            lasagne.layers.set_all_param_values(new_layers, params)

            if terminate_at == "embedding":
                return {'l_out': l_embedding, 'warmup': encoder_dict['warmup']}
            else:
                raise ValueError()

        elif freeze_at == "embedding":
            return {'l_out': l_in[0], 'warmup': 7}

        else:
            raise ValueError()

    else:
        raise ValueError()


# ---------------------------------------------------------------------------------------

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
        'warmup': warmup,
        'l_feats': l_feats
    }


def mono_lstm(feats_shape, batch_size=6, max_time=64, encoder_kwargs=None):
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
    l_cc1 = lasagne.layers.dropout(l_lstm1, p=.3)

    l_linout = lasagne.layers.DenseLayer(
        l_cc1, num_units=21, num_leading_axes=2, nonlinearity=None,
        name="l_linout")

    return {
        'l_in': [l_in],
        'l_duration': l_duration,
        'l_mask': l_mask,
        'l_linout': l_linout,
        'warmup': warmup,
        'l_feats': l_feats
    }


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
    l_cc1 = DropoutLayer(l_cc1, p=0.3)

    l_linout = lasagne.layers.DenseLayer(
        l_cc1, num_units=21, num_leading_axes=2, nonlinearity=None,
        name="l_linout")

    return {
        'l_in': [l_in],
        'l_duration': l_duration,
        'l_mask': l_mask,
        'l_linout': l_linout,
        'warmup': warmup,
        'l_feats': l_feats
    }


def fusion_lstm(skel_feats_shape, bgr_feats_shape, max_time=64, batch_size=6,
                encoder_kwargs=None):
    encoder_kwargs = encoder_kwargs or {}
    n_lstm_units = 172

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
    # l_cc1 = DropoutLayer(l_cc1, p=.3)

    l_linout = lasagne.layers.DenseLayer(
        l_cc1, num_units=21, num_leading_axes=2, nonlinearity=None,
        name="l_linout")

    return {
        'l_in': [l_in_skel, l_in_bgr],
        'l_mask': l_mask,
        'l_duration': l_duration,
        'l_linout': l_linout,
        'warmup': warmup,
        'l_feats': l_feats
    }


def transfer_lstm(*feats_shape, batch_size=6, max_time=64, encoder_kwargs=None):
    encoder_kwargs = encoder_kwargs or {}

    l_in = [lasagne.layers.InputLayer(shape=(batch_size, max_time) + s)
            for s in feats_shape]
    encoder_data = transfer_encoder(*l_in, **encoder_kwargs)
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

    # No noise here because it is included in encoder
    l_lstm1 = lasagne.layers.GRULayer(
        l_feats, num_units=172, mask_input=l_mask,
        grad_clipping=1., learn_init=True)
    l_lstm2 = lasagne.layers.GRULayer(
        l_feats, num_units=172, mask_input=l_mask,
        backwards=True, grad_clipping=1., learn_init=True)
    l_cc1 = lasagne.layers.ConcatLayer((l_lstm1, l_lstm2), axis=2)
    l_cc1 = DropoutLayer(l_cc1, p=0.3)

    l_linout = lasagne.layers.DenseLayer(
        l_cc1, num_units=21, num_leading_axes=2, nonlinearity=None,
        name="l_linout")

    return {
        'l_in': l_in,
        'l_duration': l_duration,
        'l_mask': l_mask,
        'l_linout': l_linout,
        'warmup': warmup,
        'l_feats': l_feats
    }
