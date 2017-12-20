import numpy as np
from lasagne.layers import MergeLayer
from theano import tensor as T


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


