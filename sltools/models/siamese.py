import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import MergeLayer
from lproc import chunk_load


def build_predict_fn(model_dict, batch_size):
    out = lasagne.layers.get_output(model_dict['l_linout'], deterministic=True)
    predict_batch_fn = theano.function(
        [l.input_var for l in model_dict['l_in']]
        + [model_dict['l_duration'].input_var],
        out)

    def predict_fn(sequences):
        buffers = [np.zeros(shape=(4 * batch_size,) + x.shape,
                            dtype=x.dtype) for x in next(zip(*sequences))]
        minibatch_iterator = chunk_load(sequences, buffers, batch_size, pad_last=True)

        predictions = np.concatenate([
            predict_batch_fn(*b)
            for b in minibatch_iterator], axis=0)

        return predictions[:len(sequences[0])]

    return predict_fn


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


def triplet_loss(left, middle, right, delta=1.):
    dist = (left - middle).norm(2, axis=1) ** 2 \
        - (left - right).norm(2, axis=1) ** 2
    if isinstance(left, T.TensorVariable):
        return T.maximum(dist + delta, 0)
    else:
        return np.maximum(dist + delta, 0)
