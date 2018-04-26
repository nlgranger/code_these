import logging
from random import shuffle

import numpy as np
import theano
import theano.tensor as T
import lasagne

from sltools.nn_utils import log_softmax, categorical_crossentropy_logdomain


class PosteriorModel:
    def __init__(self, build_encoder, nstates, max_len, batch_size, input_shapes=None):
        self.build_encoder = build_encoder
        self.nstates = nstates
        self.max_len = max_len
        self.batch_size = batch_size
        self.input_shapes = input_shapes

        self.input_vars = None
        self.l_in = None
        self.l_feats = None
        self.l_raw = None
        self.l_out = None
        self.warmup = None
        self._forward = None

    def __getstate__(self):
        input_shapes_arg = self.input_shapes

        state = {'build_encoder': self.build_encoder,
                 'nstates': self.nstates,
                 'max_len': self.max_len,
                 'batch_size': self.batch_size,
                 'input_shapes': input_shapes_arg}

        if self.l_out is not None:
            layers = lasagne.layers.get_all_layers(self.l_out)
            layers = layers[min(layers.index(l) for l in self.l_in) + 1:]
            state['parameters'] = lasagne.layers.get_all_param_values(layers)
        else:
            state['parameters'] = None

        return state

    def __setstate__(self, state):
        parameters = state.pop('parameters')

        self.__init__(**state)

        if parameters is not None:
            self._build()
            layers = lasagne.layers.get_all_layers(self.l_out)
            layers = layers[min(layers.index(l) for l in self.l_in) + 1:]
            lasagne.layers.set_all_param_values(layers, parameters)

    def _build(self):
        self.l_in = [lasagne.layers.InputLayer((self.batch_size, self.max_len) + shape)
                     for shape in self.input_shapes]
        self.input_vars = [l.input_var for l in self.l_in]

        encoder_data = self.build_encoder(*self.l_in)
        self.l_feats = encoder_data['l_out']

        self.warmup = encoder_data['warmup']

        if len(self.l_feats.output_shape) != 3:
            raise ValueError("The network should have an output shape like"
                             " (batch, duration, featsize).")
        if self.l_feats.output_shape[2] != self.nstates:
            logging.info("Adding one dense layer to match the output dimension")
            self.l_raw = lasagne.layers.DenseLayer(
                self.l_feats, self.nstates, num_leading_axes=2, nonlinearity=None)
        else:
            self.l_raw = self.l_feats
        self.l_out = lasagne.layers.NonlinearityLayer(
            self.l_raw, nonlinearity=log_softmax)

        self._forward = theano.function(
            self.input_vars,
            lasagne.layers.get_output(self.l_out, deterministic=True))

    def predict_logproba(self, *x):
        step = self.max_len - 2 * self.warmup

        # Chunking
        d = len(x[0])
        chunks = [(k, min(k + self.max_len, d))
                  for k in range(0, d - self.warmup, step)]
        prediction = np.zeros((d, self.nstates), dtype=theano.config.floatX)

        buffers = [np.zeros((self.batch_size, self.max_len) + shape,
                            dtype=theano.config.floatX)
                   for shape in self.input_shapes]
        c_buffer = np.zeros((self.batch_size, 2), dtype=np.int32)

        j = 0
        for i, (start, stop) in enumerate(chunks):
            c_buffer[j] = (start, stop)
            for b, xm in zip(buffers, x):
                b[j][:stop - start] = xm[start:stop]

            if j + 1 == self.batch_size or i == len(chunks) - 1:
                batch_predictions = self._forward(*buffers)[:j + 1]
                for (start_, stop_), pred in zip(c_buffer, batch_predictions):
                    warmup = self.warmup if start_ > 0 else 0
                    prediction[start_ + warmup:stop_] = pred[warmup:stop_ - start_]

            j = (j + 1) % self.batch_size

        return prediction

    def predict_proba(self, *x):
        return np.exp(self.predict_logproba(*x))

    def fit(self, X, y, weights=None, refit=False,
            n_epochs=15, l_rate=0.0002, loss="cross_entropy", updates="adam",
            callback=None):

        # Build/compile model
        if not self.l_out or not refit:
            self.input_shapes = [Xm[0].shape[1:] for Xm in X]
            self._build()

        # Setup training data
        durations = [len(seq) for seq in y]
        step = self.max_len - 2 * self.warmup
        chunks = [(seq, start, min(start + self.max_len, d))
                  for seq, d in enumerate(durations)
                  for start in range(0, d - self.warmup, step)]

        if weights is None:
            weights = np.ones((self.nstates,), dtype=np.float32)
        else:
            weights = np.asarray(weights, dtype=np.float32)
        weights = T.as_tensor_variable(weights)

        # Build/compile training routines
        durations_var = T.ivector()
        mask = T.stack(
            *[T.set_subtensor(
                T.zeros((self.max_len,), theano.config.floatX)[:durations_var[i]],
                1)
                for i in range(self.batch_size)])
        target_values = T.imatrix()

        if loss == "cross_entropy":
            linear_out = lasagne.layers.get_output(self.l_raw)
            train_predictions = log_softmax(linear_out)
            loss = T.reshape(categorical_crossentropy_logdomain(
                train_predictions.reshape((self.batch_size * self.max_len, self.nstates)),
                T.flatten(target_values)) * weights[T.flatten(target_values)],
                (self.batch_size, self.max_len)) * mask
        elif loss == "hinge":
            train_predictions = lasagne.layers.get_output(self.l_raw)
            loss = T.reshape(lasagne.objectives.multiclass_hinge_loss(
                    T.reshape(train_predictions,
                              (self.batch_size * self.max_len, self.nstates)),
                    T.flatten(target_values), delta=1)
                * weights[T.flatten(target_values)],
                (self.batch_size, self.max_len)) * mask
        else:
            raise ValueError

        loss = T.mean(loss[:, self.warmup:self.max_len - self.warmup])

        parameters = lasagne.layers.get_all_params(self.l_out, trainable=True)

        grads = theano.gradient.grad(loss, parameters)
        # grads = [lasagne.updates.norm_constraint(grad, 4, range(grad.ndim))
        #          for grad in grads]

        if updates == "nesterov":
            updates = lasagne.updates.nesterov_momentum(
                grads, parameters, learning_rate=l_rate)
        elif updates == "adam":
            updates = lasagne.updates.adam(
                grads, parameters, learning_rate=l_rate)
        else:
            raise ValueError("unsupported value for updates")

        train_fn = theano.function(
            [*self.input_vars, durations_var, target_values], loss,
            updates=updates)

        X_buffers = [np.zeros(shape=(self.batch_size, self.max_len) + shape,
                              dtype=theano.config.floatX)
                     for shape in self.input_shapes]
        d_buffer = np.zeros(shape=(self.batch_size,), dtype=np.int32)
        y_buffer = np.zeros(shape=(self.batch_size, self.max_len), dtype=np.int32)

        # Training loop
        n_batches = int(np.ceil(len(chunks) / self.batch_size))
        for epoch in range(n_epochs):
            shuffle(chunks)

            j = 0
            for i, (seq, start, stop) in enumerate(chunks):
                d_buffer[j] = stop - start
                y_buffer[j, :stop - start] = y[seq][start:stop]
                for k in range(len(self.l_in)):
                    X_buffers[k][j][:stop - start] = X[k][seq][start:stop]

                if j + 1 == self.batch_size or i == len(chunks) - 1:
                    loss = float(train_fn(*X_buffers, d_buffer, y_buffer))
                    d_buffer.fill(0)
                    y_buffer.fill(0)
                    [b.fill(0) for b in X_buffers]

                    if callback is not None:
                        callback(epoch, n_epochs, i // self.batch_size, n_batches, loss)

                j = (j + 1) % self.batch_size

        return self
