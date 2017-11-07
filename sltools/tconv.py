import theano.tensor as T
import lasagne


class TemporalConv(lasagne.layers.Layer):
    """1D convolution layer
    
    Performs 1D convolution on its input and optionally adds a bias and applies an 
    elementwise nonlinearity. 
    
    Parameters
    ----------
    
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. Must
        be a tensor of 2+`n` dimensions:
        ``(batch_size, num_input_channels, <n spatial dimensions>)``.

    num_filters : int
        An integer specifying the number of filters.

    filter_size : int
        An integer specifying the size of the filters
        
    filter_dilation: int
        An integer specifying the dilation factor of the filters. A factor of $x$ 
        corresponds to $x−1$ zeros inserted between adjacent filter elements.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a tensor of 2+`n` dimensions with shape
        ``(num_filters, num_input_channels, <n spatial dimensions>)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, <n spatial dimensions>)`` instead.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
        
    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.

    b : Theano shared variable or expression
        Variable or expression representing the biases.
    """

    def __init__(self, incoming,
                 num_filters, filter_size,
                 pad='valid', subsample=1, filter_dilation=1,
                 conv_type='regular',
                 W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 **kwargs):
        super(TemporalConv, self).__init__(incoming, **kwargs)

        if filter_size % 2 == 0:
            raise ValueError("only odd sized filter supported")

        if not isinstance(pad, int) and pad not in {'valid', 'same', 'full'}:
            raise ValueError("invalid padding value")

        if conv_type not in {'regular', 'expand', 'match'}:
            raise ValueError("invalid convolution type")

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.pad = pad
        self.subsample = subsample
        self.filter_dilation = filter_dilation
        self.conv_type = conv_type

        feat_size = self.input_shape[2]

        if conv_type == "regular":
            self.W = self.add_param(
                W, shape=(num_filters, filter_size, feat_size),
                name="W")
        elif conv_type == "expand":
            self.W = self.add_param(
                W, shape=(num_filters, filter_size),
                name="W")
        elif conv_type == "match":
            self.W = self.add_param(
                W, shape=(num_filters, filter_size),
                name="W")

        if b is None:
            self.b = None
        elif conv_type == "regular":
            self.b = self.add_param(b, (num_filters,), name="b",
                                    regularizable=False)
        elif conv_type == "expand":
            self.b = self.add_param(b, (feat_size * num_filters,), name="b",
                                    regularizable=False)
        else:  # conv_type == "match"
            self.b = self.add_param(b, (num_filters,), name="b",
                                    regularizable=False)

        self.nonlinearity = nonlinearity

    def get_output_for(self, input, **kwargs):
        batch_sz = input.shape[0]
        seq_sz = input.shape[1]
        feat_sz = input.shape[2]
        nfilters = self.num_filters
        filter_sz = self.filter_size
        subsample = self.subsample
        dilation = self.filter_dilation

        if self.pad == 'valid':
            border_mode = 0
        elif self.pad == 'full':
            border_mode = filter_sz * self.filter_dilation
        elif self.pad == 'same':
            border_mode = filter_sz // 2 * self.filter_dilation
        else:
            border_mode = self.pad

        out_seq_sz = seq_sz + 2 * border_mode - (filter_sz - 1) * dilation

        if self.conv_type == "regular":
            filters = self.W
            output = T.nnet.conv2d(
                input=input.reshape((batch_sz, 1, seq_sz, feat_sz)),
                filters=filters.reshape((nfilters, 1, filter_sz, feat_sz)),
                border_mode=(border_mode, 0),
                subsample=(subsample, 1),
                filter_dilation=(dilation, 1))
            output = output.dimshuffle(0, 2, 3, 1) \
                           .reshape((batch_sz, out_seq_sz, self.num_filters))

        elif self.conv_type == "expand":
            filters = self.W
            output = T.nnet.conv2d(
                input.reshape((batch_sz, 1, seq_sz, feat_sz)),
                filters.reshape((nfilters, 1, filter_sz, 1)),
                border_mode=(border_mode, 0),
                subsample=(subsample, 1),
                filter_dilation=(dilation, 1))
            output = output.dimshuffle(0, 2, 3, 1) \
                           .reshape((batch_sz, out_seq_sz, feat_sz * self.num_filters))

        else:  # self.conv_type == "match"
            filters = self.W
            output = T.nnet.conv2d(
                input.dimshuffle(0, 2, 1).reshape((batch_sz, feat_sz, seq_sz, 1)),
                filters.reshape((feat_sz, 1, filter_sz, 1)),
                num_groups=nfilters,
                border_mode=(border_mode, 0),
                subsample=(subsample, 1),
                filter_dilation=(dilation, 1))
            output = output.dimshuffle(0, 2, 3, 1) \
                           .reshape((batch_sz, out_seq_sz, feat_sz))

        if self.b is not None:  # TODO: separate bias for expand
            output += T.shape_padleft(self.b, 2)

        if self.nonlinearity is not None:
            output = self.nonlinearity(output)

        return output

    def get_output_shape_for(self, input_shape):
        batch_size = self.input_shape[0]
        duration = self.input_shape[1]
        feat_sz = self.input_shape[2]
        if self.conv_type == "regular":
            num_feats = self.num_filters
        elif self.conv_type == "match":
            num_feats = feat_sz
        else:
            num_feats = self.num_filters * feat_sz

        return batch_size, duration, num_feats
