import lasagne
import theano.tensor as T
from sltools.nn_utils import DurationMaskLayer

from experiments.hmmvsrnn_reco.c_models import skel_encoder


def skel_rnn(feats_shape, batch_size, max_time, encoder_kwargs):
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

    # l_d1 = lasagne.layers.dropout(l_feats, p=.3)
    l_lstm1 = lasagne.layers.GRULayer(
        l_feats, num_units=n_lstm_units, mask_input=l_mask,
        grad_clipping=1., learn_init=True, only_return_final=True)
    l_lstm2 = lasagne.layers.GRULayer(
        l_feats, num_units=n_lstm_units, mask_input=l_mask,
        backwards=True, grad_clipping=1., learn_init=True, only_return_final=True)
    l_cc1 = lasagne.layers.ConcatLayer((l_lstm1, l_lstm2), axis=1)
    # l_cc1 = lasagne.layers.dropout(l_cc1, p=.3)

    l_linout = lasagne.layers.DenseLayer(
        l_cc1, num_units=96, num_leading_axes=1,
        nonlinearity=lasagne.nonlinearities.rectify,
        name="l_linout")

    l_linout = lasagne.layers.ExpressionLayer(
        l_linout,
        lambda x: x / x.norm(2, axis=1).dimshuffle(0, 'x'))

    return {
        'l_in': [l_in],
        'l_duration': l_duration,
        'l_mask': l_mask,
        'l_linout': l_linout,
        'warmup': warmup,
        'l_feats': l_feats
    }
