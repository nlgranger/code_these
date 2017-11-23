import copy
import numpy as np
from scipy.misc import logsumexp
import pomegranate

from sltools.extra_distributions import PrecomputedDistribution
from datasets.utils import gloss2seq


class HMMRecognizer:
    def __init__(self, chains_lengths, posterior_model, labels, multiple_inputs=False):
        # Create HMM
        # idle_state -> [nlabels] * lr-words models -> idle_state

        self.chains_lengths = np.array(chains_lengths)
        self.multiple_inputs = multiple_inputs
        self.posterior = posterior_model
        self.hmm = None

        self.labels = labels
        self.label_masks = {l: np.full((self.nstates,), -np.inf)
                            for l in self.labels + [0]}
        self.label_masks[0][-1] = 0
        for i in range(self.nlabels):
            axes = sum(self.chains_lengths[:i]), sum(self.chains_lengths[:i + 1])
            self.label_masks[self.labels[i]][axes[0]:axes[1]] = 0

        self.p_s = None  # p(state)
        self.p_idle2gesture = None
        self.state2idx = None  # pomegranate -> our indexing
        self.state2label = None  # our indexing -> labels

    @property
    def nlabels(self):
        return len(self.chains_lengths)

    @property
    def nstates(self):
        return sum(self.chains_lengths) + 1

    @property
    def p_idle2idle(self):
        return 1 - self.p_idle2gesture * self.nlabels

    def predict_states(self, X):
        predictions = []
        for x in X:
            logposteriors = self.posterior.predict_logproba(*x)
            _, hmm_state_seq = self.hmm.viterbi(logposteriors - self.p_s)
            hmm_state_seq = np.array([s[0] for s in hmm_state_seq[1:]])
            predictions.append(hmm_state_seq)

        return predictions

    def predict(self, X):
        return [self.state2label[seq] for seq in self.predict_states(X)]

    def fit_priors(self, X, gloss_seqs, smoothing):
        # Compute priors
        durations = [len(x[0]) for x in X]

        n_transitions = sum([len(g) for g in gloss_seqs])
        n_idle = sum([d - sum([g[2] - g[1] for g in gseq])
                      for d, gseq in zip(durations, gloss_seqs)])
        self.p_idle2gesture = n_transitions / n_idle / self.nlabels

        self.p_s = logsumexp([
            logsumexp(self.posterior.predict_logproba(*x), axis=0)
            for x in X], axis=0)
        self.p_s -= logsumexp(self.p_s)

        self.p_s *= smoothing
        self.p_s -= logsumexp(self.p_s)

    def fit_transitions(self, X, gloss_seqs, **hmm_fit_args):
        # Train individual word models
        params = []

        for i in range(len(self.labels)):
            # Range of state indexes for this label
            axes = sum(self.chains_lengths[:i]), sum(self.chains_lengths[:i+1])

            # Compute posteriors for the states of this label
            subsgments = [(seq, start, stop)
                          for seq, gloss_seq in enumerate(gloss_seqs)
                          for l, start, stop in gloss_seq if l == self.labels[i]]
            Xw = [tuple(Xm[start:stop] for Xm in X[seq])
                  for seq, start, stop in subsgments]
            Xw = [self.posterior.predict_logproba(*x)[:, axes[0]:axes[1]]
                  for x in Xw]
            Xw = [x - self.p_s[None, axes[0]:axes[1]] for x in Xw]
            # Xw = [x - logsumexp(x, axis=1, keepdims=True) for x in Xw]

            # pseudo log-likelihoods
            params.append(
                self._fit_word_model(Xw, self.chains_lengths[i], **hmm_fit_args))

        # Create complete model
        print("loading trained parameters into the model")
        self.hmm = pomegranate.HiddenMarkovModel(None)

        states = []
        for i in range(self.nstates):
            s = pomegranate.State(PrecomputedDistribution(i, self.nstates), name=str(i))
            states.append(s)
            self.hmm.add_state(s)

        self.hmm.start.name = str(-1)
        self.hmm.end.name = str(self.nstates)
        self.hmm.add_transition(self.hmm.start, states[-1], 1)
        self.hmm.add_transition(states[-1], states[-1], self.p_idle2idle)

        for i in range(self.nlabels):
            state_offset = np.sum(self.chains_lengths[:i])
            l = self.chains_lengths[i]

            for s1, s2, p in params[i]:
                # Adjust indexes and parameters to integrate within full model
                s2 = -1 if s2 == l else s2 + state_offset
                if s1 == -1:
                    p = self.p_idle2gesture
                else:
                    s1 += state_offset

                self.hmm.add_transition(states[s1], states[s2], p)

        self.hmm.bake()

        # Build mapping between internal indexes and ours
        self.state2idx = np.array([int(s.name) for s in self.hmm.states
                                   if s.name not in {"-1", str(self.nstates)}],
                                  dtype=np.int32)
        idx2labels = np.concatenate(
            [np.full((self.chains_lengths[i],), self.labels[i])
             for i in range(self.nlabels)] + [np.array([0.0])]).astype(np.int32)
        self.state2label = np.array([idx2labels[int(s.name)]
                                     for s in self.hmm.states
                                     if int(s.name) not in {-1, self.nstates}])

    def fit_posterior(self, X_seqs, state_seqs, **posterior_fit_args):
        self.posterior.fit(X_seqs, state_seqs, **posterior_fit_args)

    def align_states(self, X_seqs, gloss_seqs, linear=False):
        if linear:  # fresh start -> hard label assignment
            seqs_duration = [len(x[0]) for x in X_seqs]
            return [self._linearstateassignment(g, d)
                    for g, d in zip(gloss_seqs, seqs_duration)]

        else:
            assert self.hmm is not None
            return [self.state2idx[self._supervized_state_alignment(f_seq, s_seq)]
                    for f_seq, s_seq in zip(X_seqs, gloss_seqs)]

    def _linearstateassignment(self, gloss_seq, seq_duration):
        """Linearly spread states within subsequences ('hard' state assignment)."""
        chains_lengths = self.chains_lengths
        idx_offsets = np.cumsum(chains_lengths) - np.array(chains_lengths)

        labels = np.full((seq_duration,), self.nstates - 1, dtype=np.int32)
        for l, start, stop in gloss_seq:
            l = self.labels.index(l)
            labels[start:stop] = \
                idx_offsets[l] \
                + np.floor(np.linspace(0, chains_lengths[l], stop - start,
                                       endpoint=False))

        return labels

    def _supervized_state_alignment(self, feat_seq, gloss_seq):
        logposterior_seq = self.posterior.predict_logproba(*feat_seq)
        logposterior_seq = self._enforce_annotations(logposterior_seq, gloss_seq)
        _, state_seq = self.hmm.viterbi(logposterior_seq - self.p_s)
        state_seq = np.array([s[0] for s in state_seq[1:]])
        return state_seq

    def _enforce_annotations(self, posterior_seq, gloss_seq):
        """Zero-out posteriors based on annotations."""
        posterior_seq = np.copy(posterior_seq)

        gloss_seq = copy.deepcopy(gloss_seq)  # insert blank label between gestures
        for i, (l, start, stop) in enumerate(gloss_seq):
            start = max(1, start)
            stop = min(len(posterior_seq) - 1, stop)
            if i < len(gloss_seq) - 1:
                stop = min(stop, gloss_seq[i+1][1] - 1)

            gloss_seq[i] = (l, start, stop)

        for i, l in enumerate(gloss2seq(gloss_seq, len(posterior_seq), 0)):
            posterior_seq[i] += self.label_masks[l]

        # Does not matter for viterbi alignment ->
        # posterior_seq -= logsumexp(posterior_seq, axis=1)[:, None]

        return posterior_seq

    @staticmethod
    def _fit_word_model(X, nstates, **kwargs):
        wmodel = pomegranate.HiddenMarkovModel(None)
        wmodel.start.name = str(-1)
        wmodel.end.name = str(nstates)

        states = [pomegranate.State(PrecomputedDistribution(s, nstates), name=str(s))
                  for s in range(nstates)]

        for s in range(nstates):
            wmodel.add_state(states[s])
            wmodel.add_transition(states[s], states[s], 0.8)

        wmodel.add_transition(wmodel.start, states[0], 1)
        for s in range(1, nstates):
            wmodel.add_transition(states[s - 1], states[s], 0.15)
        wmodel.add_transition(states[-1], wmodel.end, 0.15)
        wmodel.add_transition(states[-2], states[1], 0.05)

        for s in range(2, nstates - 1):
            wmodel.add_transition(states[s - 2], states[s], 0.05)

        wmodel.bake()

        improvement = wmodel.fit(X, **kwargs)
        if np.isnan(improvement):
            raise ValueError
        print("HMM improvement: {:2.4f}".format(improvement))

        return [(int(e[0].name), int(e[1].name), np.exp(e[2]['probability']))
                for e in wmodel.graph.edges(data=True)]
