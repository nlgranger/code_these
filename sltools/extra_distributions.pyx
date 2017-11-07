import json
cimport numpy
from pomegranate.base cimport Model


ctypedef numpy.npy_intp SIZE_t


cdef class PrecomputedDistribution(Model):
    cdef int idx

    def __init__(self, idx, d):
        self.idx = idx
        self.d = d

    def __reduce__(self):
        return self.__class__, (self.idx, self.d)

    cdef void _log_probability(self, double* symbol, double* log_probability, int n) nogil:
        log_probability[0] = symbol[self.idx]

    cdef double _vl_log_probability(self, double* symbol, int n) nogil:
        with gil:
            print("not implemented")
        return 0

    cdef double _summarize(self, double* items, double* weights,
                           SIZE_t n) nogil:
        return 0

    def to_json(self):
        return json.dumps({
            'class' :  self.__class__.__module__ + "." + self.__class__.__name__,
            'idx' : self.idx,
            'd' : self.d
        })
