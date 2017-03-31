## Equilibrium Propagation: Bridging the Gap Between Backpropagation and Energy-Based Models
For a description of the model, check the [paper](https://arxiv.org/abs/1602.05179).

The code is written in [Theano](https://github.com/Theano/Theano), the Deep Learning framework developed by [MILA](https://mila.umontreal.ca/en/).

Note that an old version of the paper was "Towards a Biologically Plausible Backprop".

## Getting started
* Download the code from GitHub:
```bash
git clone https://github.com/bscellier/Towards-a-Biologically-Plausible-Backprop
cd Towards-a-Biologically-Plausible-Backprop
```
* To train a model (with 1 hidden layer by default), run the python script:
``` bash
THEANO_FLAGS="floatX=float32, gcc.cxxflags='-march=core2'" python train_model.py
```
* Once the model is tranied, use the GUI by running the python script:
``` bash
THEANO_FLAGS="floatX=float32, gcc.cxxflags='-march=core2'" python gui.py net1
```
