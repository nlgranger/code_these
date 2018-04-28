import numpy as np
import theano
import theano.tensor as T
import lasagne
import seqtools
from sltools.nn_utils import log_softmax, adjust_length


def build_train_fn(model_dict, max_time, warmup,
                   loss_fn, updates_fn=lasagne.updates.adam):
    input_vars = [l.input_var for l in model_dict['l_in']]
    durations_var = model_dict['l_duration'].input_var
    targets = theano.tensor.imatrix()
    linout, masks = lasagne.layers.get_output(
        [model_dict['l_linout'], model_dict['l_mask']])
    loss = loss_fn(linout, targets, masks)
    loss = T.mean(loss[:, warmup:max_time - warmup])

    parameters = lasagne.layers.get_all_params(model_dict['l_linout'], trainable=True)
    l_rate_var = T.scalar('l_rate')
    updates = updates_fn(loss, parameters, learning_rate=l_rate_var)

    return theano.function(
        [*input_vars, targets, durations_var, l_rate_var],
        loss, updates=updates)


def build_predict_fn(model_dict, batch_size, max_time, nworkers=2):
    warmup = model_dict['warmup']
    l_in = model_dict['l_in']
    l_duration = model_dict['l_duration']
    l_out = lasagne.layers.ExpressionLayer(model_dict['l_linout'], log_softmax)

    out_var = lasagne.layers.get_output(l_out, deterministic=True)
    predict_batch_fn = theano.function(
        [l.input_var for l in l_in] + [l_duration.input_var], out_var)

    def predict_fn(feature_sequences):
        durations = np.array([len(s) for s in feature_sequences[0]])
        step = max_time - 2 * warmup

        # turn sequences
        chunks = [(i, k, min(d, k + max_time))
                  for i, d in enumerate(durations)
                  for k in range(0, d - warmup, step)]
        chunked_sequences = []
        for feat in feature_sequences:
            def get_chunk(i, t1, t2, feat_=feat):
                return adjust_length(feat_[i][t1:t2], size=max_time, pad=0)

            chunked_sequences.append(seqtools.starmap(get_chunk, chunks))
        chunked_sequences.append([np.int32(t2 - t1) for _, t1, t2 in chunks])
        chunked_sequences = seqtools.collate(chunked_sequences)

        # turn into minibatches
        null_sample = chunked_sequences[0]
        n_features = len(null_sample)

        def collate(b):
            return [np.array([b[i][c] for i in range(batch_size)])
                    for c in range(n_features)]

        minibatches = seqtools.batch(
            chunked_sequences, batch_size, pad=null_sample, collate_fn=collate)
        minibatches = seqtools.prefetch(
            minibatches, nworkers=nworkers, max_buffered=nworkers * 5)

        # process
        batched_predictions = seqtools.starmap(predict_batch_fn, minibatches)
        batched_predictions = seqtools.add_cache(batched_predictions)
        chunked_predictions = seqtools.unbatch(batched_predictions, batch_size)

        # recompose
        out = [np.empty((d,) + l_out.output_shape[2:], dtype=np.float32)
               for d in durations]

        for v, (s, start, stop) in zip(chunked_predictions, chunks):
            skip = warmup if start > 0 else 0
            out[s][start + skip:stop] = v[skip:stop - start]

        return out

    return predict_fn
