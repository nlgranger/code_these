import theano.tensor as T
import lasagne
from lproc import SerializableFunc
from sltools.tconv import TemporalConv
from sltools.nn_utils import DurationMaskLayer


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


@SerializableFunc
def build_lstm(feats_shape, batch_size=6, max_time=64):
    n_lstm_units = 172

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, max_time) + feats_shape, name="l_in")
    encoder_data = build_encoder(l_in)
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
        l_d1, num_units=n_lstm_units, mask_input=l_mask,
        backwards=True, grad_clipping=1., learn_init=True)
    l_cc1 = lasagne.layers.ConcatLayer((l_lstm1, l_lstm2), axis=2)
    l_cc1 = lasagne.layers.dropout(l_cc1, p=.3)

    l_linout = lasagne.layers.DenseLayer(
        l_cc1, num_units=21, num_leading_axes=2, nonlinearity=None,
        name="l_linout")

    return {
        'l_in': l_in,
        'l_duration': l_duration,
        'l_mask': l_mask,
        'l_linout': l_linout,
        # 'l_out': l_lstm,
        'warmup': warmup,
        'l_feats': l_feats
    }
