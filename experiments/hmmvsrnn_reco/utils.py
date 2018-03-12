import os
import shelve
import lasagne
from sltools.models.rnn import build_predict_fn
from experiments.hmmvsrnn_reco.b_preprocess import skel_feat_seqs, bgr_feat_seqs

from experiments.hmmvsrnn_reco.a_data import tmpdir


def autoreload_feats(modality, **kwargs):
    from experiments.hmmvsrnn_reco.b_preprocess import skel_feat_seqs, bgr_feat_seqs

    if modality == "skel":  # Skeleton end-to-end
        return [skel_feat_seqs]

    elif modality == "bgr":  # BGR end-to-end
        return [bgr_feat_seqs]

    elif modality == "fusion":  # Fusion end-to-end
        return [skel_feat_seqs, bgr_feat_seqs]

    elif modality == "transfer":  # Transfer
        from experiments.hmmvsrnn_reco.b_preprocess import transfer_feats

        transfer_from = kwargs['transfer_from']
        freeze_at = kwargs['freeze_at']

        if freeze_at == 'input':
            report = shelve.open(os.path.join(tmpdir, transfer_from))
            modality = report['meta']['modality']
            return autoreload_feats(modality)

        else:
            return transfer_feats(transfer_from, freeze_at)

    else:
        raise ValueError


def reload_best_rnn(report):
    modality = report['meta']['modality']

    best_epoch = sorted([(r['val_scores']['jaccard'], e)
                         for e, r in report.items()
                         if e.startswith("epoch")
                         and "params" in r.keys()])[-1][1]

    if modality == "skel":  # Skeleton end-to-end
        from experiments.hmmvsrnn_reco.c_models import skel_lstm
        model_dict = skel_lstm(feats_shape=skel_feat_seqs[0][0].shape,
                               **report['model_args'])

    elif modality == "bgr":  # BGR end-to-end
        from experiments.hmmvsrnn_reco.c_models import bgr_lstm
        model_dict = bgr_lstm(feats_shape=bgr_feat_seqs[0][0].shape,
                              **report['model_args'])

    elif modality == "fusion":  # Fusion end-to-end
        from experiments.hmmvsrnn_reco.c_models import fusion_lstm
        model_dict = fusion_lstm(skel_feats_shape=skel_feat_seqs[0][0].shape,
                                 bgr_feats_shape=bgr_feat_seqs[0][0].shape,
                                 **report['model_args'])

    else:
        raise ValueError('unexpected modality type')

    # Reload parameters
    params = report[best_epoch]['params']
    all_layers = lasagne.layers.get_all_layers(model_dict['l_linout'])
    lasagne.layers.set_all_param_values(all_layers, params)

    # Compile
    predict_fn = build_predict_fn(
        model_dict,
        report['model_args']['batch_size'],
        report['model_args']['max_time'])

    return best_epoch, model_dict, predict_fn


def reload_best_hmm(report):
    epochs = sorted([e for e in report.keys() if e.startswith('epoch')])
    scores = [report[e]['val_scores']['jaccard'] for e in epochs]
    i = scores.index(max(scores))
    best_epoch = epochs[i]

    recognizer = report[best_epoch]['model']
    if i > 0:
        previous_recognizer = report[epochs[i - 1]]['model']
    else:
        previous_recognizer = None

    return best_epoch, recognizer, previous_recognizer
