from random import shuffle
from functools import partial
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lproc import rmap, add_cache, chunk_load
from sltools.nn_utils import log_softmax, adjust_length


class build_train_fn:
    def __init__(self, layers, batch_size, max_len, warmup,
                 loss_fn, updates_fn=lasagne.updates.adam):
        self.multiple_inputs = not isinstance(layers['l_in'], lasagne.layers.Layer)
        l_in = layers['l_in'] if self.multiple_inputs else [layers['l_in']]
        l_duration = layers['l_duration']
        l_mask = layers['l_mask']
        l_linout = layers['l_linout']
        self.max_len = max_len
        self.batch_size = batch_size
        self.warmup = warmup

        input_vars = [l.input_var for l in l_in]
        durations_var = l_duration.input_var
        targets = theano.tensor.imatrix()
        linout, masks = lasagne.layers.get_output([l_linout, l_mask])
        loss = loss_fn(linout, targets, masks)
        loss = T.mean(loss[:, self.warmup:self.max_len - self.warmup])

        parameters = lasagne.layers.get_all_params(l_linout, trainable=True)
        l_rate_var = T.scalar('l_rate')
        grads = theano.grad(loss, parameters)
        grads = [T.clip(grad, -0.01, 0.01) for grad in grads]  # TODO: as parameter
        updates = updates_fn(grads, parameters, learning_rate=l_rate_var)

        self.train_batch_fn = theano.function(
            [*input_vars, durations_var, targets, l_rate_var],
            loss, updates=updates)

    def __call__(self, X, y, l_rate, n_epochs=1, batch_callback=lambda *args: None):
        if not self.multiple_inputs:
            X = rmap(lambda x: (x,), X)
        X = add_cache(X)

        n_inputs = len(X[0])

        # Prepare data
        input_shapes = [x[0].shape for x in X[0]]

        step = self.max_len - 2 * self.warmup
        durations = [len(seq) for seq in y]
        chunks = [(i, k, min(k + self.max_len, d))
                  for i, d in enumerate(durations)
                  for k in range(0, d, step)]

        # Prepare data sources and buffers
        sources = []
        for i in range(n_inputs):  # features
            def get_chunk(c, modality):
                r, start, stop = c
                return adjust_length(X[r][modality][start:stop], self.max_len)

            s = rmap(partial(get_chunk, modality=i), chunks)
            sources.append(s)

        sources.append(rmap(lambda c: c[2] - c[1], chunks))  # durations
        sources.append(rmap(lambda c:  # labels
                            adjust_length(y[c[0]][c[1]:c[2]], self.max_len),
                            chunks))

        buffers = [np.zeros(shape=(self.batch_size, self.max_len) + shape,
                            dtype=theano.config.floatX) for shape in input_shapes]
        buffers.append(np.zeros(shape=(self.batch_size,), dtype=np.int32))
        buffers.append(np.zeros(shape=(self.batch_size, self.max_len), dtype=np.int32))

        # Epoch loop
        n_batches = len(chunks) // self.batch_size
        for epoch in range(n_epochs):
            shuffle(chunks)
            batch_iter = chunk_load(sources, buffers, self.batch_size)
            for i, minibatch in enumerate(batch_iter):
                if len(minibatch[0]) < self.batch_size:
                    continue

                last_loss = self.train_batch_fn(*minibatch, l_rate)
                batch_callback(i, n_batches, float(last_loss))

        return self


class build_predict_fn:
    def __init__(self, layers, batch_size, max_time, nlabels, warmup):
        self.multiple_inputs = not isinstance(layers['l_in'], lasagne.layers.Layer)
        l_in = layers['l_in'] if self.multiple_inputs else [layers['l_in']]
        self.n_inputs = len(l_in)
        l_duration = layers['l_duration']
        l_linout = layers['l_linout']
        self.max_time = max_time
        self.batch_size = batch_size
        self.warmup = warmup

        linout = lasagne.layers.get_output(l_linout, deterministic=True)
        out = T.exp(log_softmax(linout))
        self.nlabels = nlabels
        self.predict_fn = theano.function(
            [l.input_var for l in l_in] + [l_duration.input_var],
            out)

    def __call__(self, X):
        if not self.multiple_inputs:
            X = rmap(lambda x_: (x_,), X)

        input_shapes = [x[0].shape for x in X[0]]

        step = self.max_time - 2 * self.warmup

        # Chunking
        durations = [len(seq[0]) for seq in X]
        chunks = [(i, k, min(k + self.max_time, d))
                  for i, d in enumerate(durations)
                  for k in range(0, d - self.warmup, step)]
        predictions = [np.zeros((d, self.nlabels), dtype=theano.config.floatX)
                       for d in durations]

        # Memory buffers
        X_buffers = [np.zeros(shape=(self.batch_size, self.max_time) + shape,
                              dtype=theano.config.floatX)
                     for shape in input_shapes]
        d_buffer = np.zeros((self.batch_size,), dtype=np.int32)
        c_buffer = np.zeros((self.batch_size, 3), dtype=np.int32)

        # Processing loop
        j = 0
        for i, (seq, start, stop) in enumerate(chunks):
            d_buffer[j] = stop - start
            c_buffer[j] = (seq, start, stop)
            for b, x in zip(X_buffers, X[seq]):
                b[j][:stop - start] = x[start:stop]

            if j + 1 == self.batch_size or i == len(chunks) - 1:
                batch_predictions = self.predict_fn(*X_buffers, d_buffer)[:j + 1]
                for (seq_, start_, stop_), pred in zip(c_buffer, batch_predictions):
                    warmup = self.warmup if start_ > 0 else 0
                    predictions[seq_][start_ + warmup:stop_] = \
                        pred[warmup:stop_ - start_]

            j = (j + 1) % self.batch_size

        return predictions
