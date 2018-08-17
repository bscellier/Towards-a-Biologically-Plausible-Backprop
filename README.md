# Equilibrium Propagation
Links to the papers:

* [Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation](https://www.frontiersin.org/articles/10.3389/fncom.2017.00024/full)

* [Equivalence of Equilibrium Propagation and Recurrent Backpropagation](https://arxiv.org/abs/1711.08416)

* [Generalization of Equilibrium Propagation to Vector Field Dynamics](https://arxiv.org/abs/1808.04873)

The code is written in [Theano](https://github.com/Theano/Theano), the Deep Learning framework which was developed by [MILA](https://mila.umontreal.ca/en/).

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
* Once the model is trained, use the GUI by running the python script:
``` bash
THEANO_FLAGS="floatX=float32, gcc.cxxflags='-march=core2'" python gui.py net1
```
