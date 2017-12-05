import os
import shelve
import pickle as pkl
import numpy as np
import lasagne
import theano.tensor as T
from lasagne.layers import MergeLayer
from sltools.nn_utils import DurationMaskLayer
from sltools.tconv import TemporalConv
from experiments.ch14_skel.a_data import tmpdir as skel_tmpdir
from experiments.ch14_skel.c_models import build_lstm as build_skel_lstm


def pairwise_metric(x1, x2):
    return 1 - T.sum(x1 * x2, axis=1) \
           / (x1.norm(2, axis=1) + 0.0001) \
           / (x2.norm(2, axis=1) + 0.0001)
    # return (x1 - x2).norm(2, axis=1)
    # return T.sum(x1 * x2, axis=1)


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


class SiameseInputLayer(MergeLayer):
    def __init__(self, incomings, batch_axis=0, **kwargs):
        """Aggregates n inputs of identical shapes by interleaving the elements.

        Example: [a1, b1, c1], [a2, b2, c2] -> [a1, a2, b1, b2, c1, c2]
        """
        super(SiameseInputLayer, self).__init__(incomings, **kwargs)
        self.batch_axis = batch_axis

    def get_output_for(self, inputs, **kwargs):
        n_inputs = len(inputs)
        input_shape = [n_inputs * s if i == self.batch_axis else s
                       for i, s in enumerate(inputs[0].shape)]
        out = T.zeros(input_shape, dtype=inputs[0].dtype)
        for i in range(n_inputs):
            slices = [slice(i, None, n_inputs) if j == self.batch_axis else slice(None)
                      for j in range(len(input_shape))]
            out = T.set_subtensor(out[slices], inputs[i])
        return out

    def get_output_shape_for(self, input_shapes):
        n_inputs = len(input_shapes)
        return tuple([n_inputs * s if i == self.batch_axis else s
                      for i, s in enumerate(input_shapes[0])])


# class SiameseOutputLayer(Layer):
#     def __init__(self, incoming, batch_axis=0, concat_axis=-1, n=2, **kwargs):
#         """Aggregates successive inputs from siamese net.
#
#         Example: [a1, a2, b1, b2, c1, c2] -> [[a1 a2], [b1 b2], [c1 c2]]
#         """
#         super(SiameseOutputLayer, self).__init__(incoming, **kwargs)
#         self.batch_axis = batch_axis
#         self.concat_axis = concat_axis
#         self.n = n
#         self.ndims = len(incoming.output_shape)
#
#     def get_output_for(self, input, **kwargs):
#         slices = [(slice(None),) * (self.batch_axis - 1) + (slice(i, None, self.n),)
#                   for i in range(self.n)]
#         return T.concatenate([input[s] for s in slices], axis=self.concat_axis)
#
#     def get_output_shape_for(self, input_shape):
#         return tuple([s // self.n if d == self.batch_axis
#                       else s * self.n if d == (self.concat_axis % len(input_shape))
#                       else s
#                       for d, s in enumerate(input_shape)])


class MaskedMean(MergeLayer):
    def __init__(self, incomings, axis=None, **kwargs):
        super(MaskedMean, self).__init__(incomings, **kwargs)
        self.axis = axis

    def get_output_for(self, inputs, **kwargs):
        feats, mask = inputs
        mask = T.cast(mask, 'floatX')
        return T.sum(feats * mask, axis=self.axis) / mask.sum(axis=self.axis)[:, None]

    def get_output_shape_for(self, input_shapes):
        if self.axis is None:
            return tuple()
        else:
            return input_shapes[0][:self.axis] + input_shapes[0][self.axis + 1:]


def masked_mean(incoming, mask, axis=None):
    return MaskedMean((incoming, mask), axis)


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
    l_in_left = lasagne.layers.InputLayer(
        shape=(batch_size, max_time) + feat_shape, name="l_in_left")
    l_in_right = lasagne.layers.InputLayer(
        shape=(batch_size, max_time) + feat_shape, name="l_in_right")
    l_in = SiameseInputLayer([l_in_left, l_in_right])

    l_durations_left = lasagne.layers.InputLayer(
        (batch_size,), input_var=T.ivector(), name="l_duration_left")
    l_duration_right = lasagne.layers.InputLayer(
        (batch_size,), input_var=T.ivector(), name="l_duration_right")
    l_duration = SiameseInputLayer([l_durations_left, l_duration_right])
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
    params = pretrained_skel_params(feat_shape)
    all_layers = lasagne.layers.get_all_layers(l_cc1)
    i1 = all_layers.index(l_in)
    i2 = all_layers.index(l_cc1)
    lasagne.layers.set_all_param_values(all_layers[i1+1:i2], params)

    l_linout = lasagne.layers.ExpressionLayer(
        l_f4, lambda X: pairwise_metric(X[0::2], X[1::2]),
        output_shape=(batch_size,))

    return {
        'l_in': [l_in_left, l_in_right],
        'l_duration': [l_durations_left, l_duration_right],
        'l_linout': l_linout,
        'warmup': warmup,
        'l_feats': encoder_data['l_out']
    }
