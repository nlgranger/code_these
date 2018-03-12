from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [Extension("sltools.extra_distributions",
                        ["sltools/extra_distributions.pyx"])]

setup(
    name='rnn_hmm_gesture_iconip17',
    version='0.0.1.dev0',
    description='The source code for the paper: Comparing Hybrid NN-HMM and RNN for '
                'temporal modeling in gesture models',
    packages=['ch14dataset', 'sltools'],
    install_requires=['numpy', 'scipy', 'scikit-learn', 'theano', 'pomegranate',
                      'lasagne', 'matplotlib', 'cython', 'lproc', 'seqtools'],
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
    url='https://github.com/pixelou/SLTools',
    license='MPL2',
    author='Nicolas Granger',
    author_email='nicolas.granger@telecom-sudparis.eu')
