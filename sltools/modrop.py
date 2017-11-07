import theano
import lasagne
from theano.gof import MissingInputError
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class Modropped(lasagne.layers.Layer):
    def __init__(self, incoming, mask, **kwargs):
        super(Modropped, self).__init__(incoming, **kwargs)
        self.mask = mask

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            return input
        else:
            return input * self.mask


def modrop(incomings, p, snrg=None):
    p = T.as_tensor_variable(p).astype(theano.config.floatX)
    if p.ndim == 0:
        p = T.ones((len(incomings) + 1,)) * p
    if p.ndim == 1:
        try:
            l = p.shape[0].eval()
        except MissingInputError:
            raise ValueError('The shape of p must be known')
        if l == len(incomings):
            p = T.concatenate([p, [1 - p.sum()]])

    snrg = snrg or RandomStreams()
    mask = 1 - T.cast(snrg.multinomial(pvals=p), theano.config.floatX)
    mask *= 1 / (p + p[-1])

    return tuple([Modropped(incomings[i], mask[i]) for i in range(len(incomings))])
