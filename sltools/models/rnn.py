import random
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lproc import rmap, chunk_load
from sltools.nn_utils import log_softmax, adjust_length


def as_chunks(sequence, chunks, max_time):
    return rmap(lambda c: adjust_length(sequence[c[0]][c[1]:c[2]], max_time),
                chunks)


def from_chunks(values, durations, chunks):
    result = [np.empty((d,) + values[0].shape[1:], dtype=values[0].dtype)
              for d in durations]
    for v, (s, start, stop) in zip(values, chunks):
        result[s][start:stop] = v[:stop - start]

    return result


def build_train_fn(layers, max_len, warmup,
                   loss_fn, updates_fn=lasagne.updates.adam):
        input_vars = [l.input_var for l in layers['l_in']]
        durations_var = layers['l_duration'].input_var
        targets = theano.tensor.imatrix()
        linout, masks = lasagne.layers.get_output([layers['l_linout'], layers['l_mask']])
        loss = loss_fn(linout, targets, masks)
        loss = T.mean(loss[:, warmup:max_len - warmup])

        parameters = lasagne.layers.get_all_params(layers['l_linout'], trainable=True)
        l_rate_var = T.scalar('l_rate')
        grads = theano.grad(loss, parameters)
        grads = [T.clip(grad, -0.01, 0.01) for grad in grads]  # TODO: as parameter
        updates = updates_fn(grads, parameters, learning_rate=l_rate_var)

        return theano.function(
            [*input_vars, targets, durations_var, l_rate_var],
            loss, updates=updates)


def seqs2batches(inputs, targets, batch_size, max_time, warmup,
                 shuffle=False, drop_last=False):
    durations = np.array([len(s) for s in inputs[0]])
    step = max_time - 2 * warmup
    chunks = [(i, k, min(k + max_time, d))
              for i, d in enumerate(durations)
              for k in range(0, d, step)]

    if shuffle:
        random.shuffle(chunks)

    chunked_sequences = [as_chunks(s, chunks, max_time)
                         for s in inputs + [targets]]
    chunks_durations = np.array([stop - start for _, start, stop in chunks],
                               dtype=np.int32)

    buffers = [np.zeros(shape=(4 * batch_size,) + x.shape,
                        dtype=x.dtype) for x in next(zip(*chunked_sequences))]
    buffers.append(np.zeros((4 * batch_size,), dtype=np.int32))
    return chunk_load(
        chunked_sequences + [chunks_durations], buffers, batch_size, drop_last)


def build_predict_fn(model_dict, batch_size, max_time):
    linout = lasagne.layers.get_output(model_dict['l_linout'], deterministic=True)
    out = T.exp(log_softmax(linout))
    predict_batch_fn = theano.function(
        [l.input_var for l in model_dict['l_in']]
        + [model_dict['l_duration'].input_var],
        out)

    def predict_fn(sequences):
        durations = np.array([len(s) for s in sequences[0]])
        step = max_time - 2 * model_dict['warmup']
        chunks = [(i, k, min(k + max_time, d))
                  for i, d in enumerate(durations)
                  for k in range(0, d - model_dict['warmup'], step)]

        chunks_durations = np.array([stop - start for _, start, stop in chunks],
                                   dtype=np.int32)
        chunked_sequences = [as_chunks(s, chunks, max_time) for s in sequences]

        buffers = [np.zeros(shape=(4 * batch_size,) + x.shape,
                            dtype=x.dtype) for x in next(zip(*chunked_sequences))]
        buffers.append(np.zeros((4 * batch_size,), dtype=np.int32))
        minibatch_iterator = chunk_load(
            chunked_sequences + [chunks_durations], buffers, batch_size, False)

        chunked_predictions = np.concatenate([
            predict_batch_fn(*b)
            for b in minibatch_iterator], axis=0)

        predictions = from_chunks(chunked_predictions, durations, chunks)

        return predictions

    return predict_fn
