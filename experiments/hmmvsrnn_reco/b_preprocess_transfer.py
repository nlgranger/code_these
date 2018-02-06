import os
import shelve
import pickle as pkl
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lproc import chunk_load

from sltools.nn_utils import as_chunks, from_chunks

from experiments.hmmvsrnn_reco.a_data import tmpdir
from experiments.hmmvsrnn_reco.b_preprocess import skel_feat_seqs, bgr_feat_seqs
from experiments.hmmvsrnn_reco.c_models import skel_lstm, bgr_lstm, fusion_lstm


def feature_extractor(experiment_name, feature_type,
                      max_time=128, batch_size=16, encoder_kwargs=None):
    report = shelve.open(os.path.join(tmpdir, experiment_name))

    if feature_type.startswith("rnn"):
        best_epoch = sorted([(r['val_scores']['jaccard'], e)
                             for e, r in report.items()
                             if e.startswith("epoch")
                             and "params" in r.keys()])[-1][1]
        epoch_report = report[best_epoch]

        if feature_type == "rnn_skel":
            model_dict = skel_lstm(feats_shape=skel_feat_seqs[0][0].shape,
                                   batch_size=batch_size, max_time=max_time,
                                   encoder_kwargs=encoder_kwargs)
        elif feature_type == "rnn_bgr":
            model_dict = bgr_lstm(feats_shape=bgr_feat_seqs[0][0].shape,
                                  batch_size=batch_size, max_time=max_time)
        elif feature_type == "rnn_fusion":
            model_dict = fusion_lstm(skel_feats_shape=skel_feat_seqs[0][0].shape,
                                     bgr_feats_shape=bgr_feat_seqs[0][0].shape,
                                     batch_size=batch_size, max_time=max_time)
        else:
            raise ValueError("Unsuported feature type")

        params = epoch_report['params']
        all_layers = lasagne.layers.get_all_layers(model_dict['l_linout'])
        lasagne.layers.set_all_param_values(all_layers, params)

        return model_dict

    elif feature_type.startswith("hmm"):
        best_score = 0
        recognizer = None
        for e in sorted([e for e in report.keys() if e.startswith('epoch')]):
            r = report[e]

            if r['val_report']['jaccard'] > best_score:
                best_score = r['val_report']['jaccard']
                recognizer = r['model']

        model_dict = {
            'l_in': recognizer.posterior.l_in,
            'l_feats': recognizer.posterior.l_feats,
            'warmup': recognizer.posterior.warmup
        }

        return model_dict

    elif feature_type.startswith("post"):
        best_score = 0
        recognizer = None
        for e in sorted([e for e in report.keys() if e.startswith('epoch')]):
            r = report[e]

            if r['val_report']['jaccard'] > best_score:
                best_score = r['val_report']['jaccard']
                recognizer = r['model']

        l_post = lasagne.layers.NonlinearityLayer(
            recognizer.l_raw,
            T.exp)

        model_dict = {
            'l_in': recognizer.posterior.l_in,
            'l_feats': l_post,
            'warmup': recognizer.posterior.warmup
        }

        return model_dict


def feature_extractor_fn(model_dict, max_time=128, batch_size=16,):
    feats = lasagne.layers.get_output(model_dict['l_feats'], deterministic=True)
    predict_batch_fn = theano.function(
        [l.input_var for l in model_dict['l_in']],
        feats)

    def predict_fn(sequences):
        durations = np.array([len(s) for s in sequences[0]])
        step = max_time - 2 * model_dict['warmup']
        chunks = [(i, k, min(k + max_time, d))
                  for i, d in enumerate(durations)
                  for k in range(0, d - model_dict['warmup'], step)]

        chunked_sequences = [as_chunks(s, chunks, max_time) for s in sequences]

        buffers = [np.zeros(shape=(4 * batch_size,) + x.shape,
                            dtype=x.dtype) for x in next(zip(*chunked_sequences))]
        minibatch_iterator = chunk_load(
            chunked_sequences, buffers, batch_size,
            drop_last=False, pad_last=True)

        chunked_predictions = np.concatenate([
            predict_batch_fn(*b)
            for b in minibatch_iterator], axis=0)

        predictions = from_chunks(chunked_predictions, durations, chunks)

        return predictions

    return predict_fn


def transfer_features(
        experiment_name, feature_type,
        max_time=128, batch_size=16, encoder_kwargs=None):
    cache_file = os.path.join(tmpdir, experiment_name +
                              "_" + feature_type + "_precomputed_feats.pkl")

    if feature_type.endswith("skel"):
        X = [skel_feat_seqs]
    elif feature_type.endswith("skel"):
        X = [bgr_feat_seqs]
    elif feature_type.endswith("fusion"):
        X = [skel_feat_seqs, bgr_feat_seqs]
    else:
        raise ValueError

    if not os.path.exists(cache_file):
        # generate cache feature
        model_dict = feature_extractor(
            experiment_name, feature_type,
            max_time=max_time, batch_size=batch_size, encoder_kwargs=encoder_kwargs)
        compute_feats = feature_extractor_fn(model_dict, max_time, batch_size)

        feats = compute_feats(X)
        with open(os.path.join(cache_file), 'wb') as f:
            pkl.dump(feats, f)

        return feats

    else:
        with open(os.path.join(cache_file), 'rb') as f:
            return pkl.load(f)
