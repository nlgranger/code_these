import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, GRULayer, \
    BatchNormLayer, DropoutLayer, ConcatLayer, NonlinearityLayer
from lasagne.nonlinearities import leaky_rectify, sigmoid, rectify, tanh
import seqtools

from sltools.nn_utils import DurationMaskLayer, adjust_length
from sltools.tconv import TemporalConv


def skel_encoder(l_in, tconv_sz, filter_dilation, num_tc_filters, dropout):
    warmup = (tconv_sz * filter_dilation) // 2

    l1 = DenseLayer(
        l_in, num_units=256,
        num_leading_axes=2,
        nonlinearity=None)
    l1 = BatchNormLayer(l1, axes=(0, 1))
    l1 = NonlinearityLayer(l1, leaky_rectify)

    d1 = DropoutLayer(l1, p=dropout)

    l2 = DenseLayer(
        d1, num_units=256,
        num_leading_axes=2,
        nonlinearity=None)
    l2 = BatchNormLayer(l2, axes=(0, 1))
    l2 = NonlinearityLayer(l2, leaky_rectify)

    d2 = DropoutLayer(l2, p=dropout)

    l3 = TemporalConv(d2, num_filters=num_tc_filters, filter_size=tconv_sz,
                      filter_dilation=filter_dilation, subsample=4, pad='same',
                      conv_type='regular',
                      nonlinearity=None)
    l3 = BatchNormLayer(l3, axes=(0, 1))
    l3 = NonlinearityLayer(l3, leaky_rectify)

    return {
        'l_out': l3,
        'warmup': warmup
    }


def skel_rnn(feats_shape, batch_size, max_time, encoder_kwargs):
    encoder_kwargs = encoder_kwargs or {}
    subsample = 4  # todo: detect value

    l_in = InputLayer(
        shape=(batch_size, max_time) + feats_shape, name="l_in")
    encoder_data = skel_encoder(l_in, **encoder_kwargs)
    l_feats = encoder_data['l_out']
    l_feats = DropoutLayer(l_feats, p=encoder_kwargs['dropout'])

    warmup = encoder_data['warmup']

    # LSTM layers
    durations = T.ivector()
    l_duration = InputLayer(
        (batch_size,), input_var=durations,
        name="l_duration")
    l_feat_duration = NonlinearityLayer(l_duration, lambda x: x // subsample)
    l_mask = DurationMaskLayer(
        l_feat_duration, max_time,
        name="l_mask")

    l_lstm1 = GRULayer(
        l_feats, num_units=128, mask_input=l_mask,
        grad_clipping=1., learn_init=True, only_return_final=True)
    l_lstm2 = GRULayer(
        l_feats, num_units=128, mask_input=l_mask,
        backwards=True, grad_clipping=1., learn_init=True, only_return_final=True)
    l_cc1 = ConcatLayer((l_lstm1, l_lstm2), axis=1)
    # l_cc1 = dropout(l_cc1, p=.3)

    l_linout = DenseLayer(
        l_cc1, num_units=72,
        nonlinearity=tanh,
        name="l_linout")

    # l_linout = ExpressionLayer(
    #     l_linout,
    #     lambda x: x / (x.norm(2, axis=1).dimshuffle(0, 'x') + 0.01))

    return {
        'l_in': [l_in],
        'l_duration': l_duration,
        'l_mask': l_mask,
        'l_linout': l_linout,
        'warmup': warmup,
        'l_feats': l_feats
    }


def build_predict_fn(model_dict, batch_size, max_time):
    out = lasagne.layers.get_output(model_dict['l_linout'], deterministic=True)
    predict_batch_fn = theano.function(
        [l.input_var for l in model_dict['l_in']]
        + [model_dict['l_duration'].input_var],
        out)

    def predict_fn(sequences, durations):
        # clip duration and pad
        sequences = [
            seqtools.smap(lambda s: adjust_length(s, max_time), feat_seqs)
            for feat_seqs in sequences]
        durations = np.fmin(durations, max_time)

        # batch
        batches = [
            seqtools.batch(
                feat_seqs, batch_size,
                drop_last=False, pad=np.zeros_like(feat_seqs[0]),
                collate_fn=np.stack)
            for feat_seqs in sequences]
        batches.append(
            seqtools.batch(
                durations, batch_size,
                drop_last=False, pad=np.zeros([1], dtype=np.int32),
                collate_fn=np.array))

        batches = seqtools.collate(batches)

        batch_iterator = seqtools.prefetch(batches, nworkers=2, max_buffered=20)

        predictions = np.concatenate([
            predict_batch_fn(*b)
            for b in batch_iterator], axis=0)

        return predictions[:len(sequences[0])]

    return predict_fn
