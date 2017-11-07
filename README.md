# Comparing Hybrid NN-HMM and RNN for temporal modeling in gesture recognition

This repository contains the source code for 'Comparing Hybrid NN-HMM and RNN for
temporal modeling in gesture recognition'.


## Installation

For GPU support, install CUDA and CuDNN on your machine (highly recommended).

To manage dependencies, we suggest using 
[conda](https://conda.io/docs/install/quick.html) with the following commands:

```bash
conda create -n experiments python=3.6
source activate experiments
conda install -c conda-forge scipy mkl pygpu cython matplotlib git jupyter notebook
pip install git+https://github.com/Theano/Theano.git@be022807cc9e3f8115a01ad2343c57f819f13ad9
pip install git+https://github.com/Lasagne/Lasagne.git@b1e5bc468a2fbc5e5d026f6d1c6170b80e8be224
pip install git+https://github.com/jmschrei/pomegranate.git@089dcb5bbd36d4479352ad277b3913889bed1bac
pip install git+https://github.com/pixelou/LazyProc
```

Download this project and compile the cython module:

```bash
git clone https://github.com/nlgranger/rnn_hmm_gesture_iconip17.git
cd rnn_hmm_gesture_iconip17
python setup.py build_ext --inplace
```

Using your credentials for http://sunai.uoc.edu/chalearnLAP, download and deflate the 
dataset archives, then transform the data in binary format:

```bash
mkdir dataset/data && cd dataset/data
bash retrieve.sh ftpuser ftppassword testarchivepassword
python ../compile.py
cd ../..
``` 

## Run experiments

To run the experiments, you need to run the following files:

- [experiments/ch14_skel/a_data.py]
- [experiments/ch14_skel/b_preprocess.py]
- [experiments/ch14_skel/c_model.py]
- [experiments/d_train_hmm.py] or [experiments/d_train_rnn.py]

Most of these files have setting variables at the top, so have a look at the ode and 
modify to your needs.

Then use the analyse the results with the ipython notebooks 
[experiments/e_analyze_hmm.py] or [experiments/e_analyze_rnn.py].


## License

Unless specified otherwise, all files in the project are distributed under the
Mozilla Public License Version 2.0. See [LICENSE.txt](LICENSE.txt) for more information.


## Contact

Nicolas Granger [nicolas.granger@telecom-sudparis.eu](mailto:nicolas.granger@telecom-sudparis.eu)
