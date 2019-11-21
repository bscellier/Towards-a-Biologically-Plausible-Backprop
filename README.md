# Equilibrium Propagation

Original paper:
* [Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation](https://www.frontiersin.org/articles/10.3389/fncom.2017.00024/full)

Equilibrium Propagation (EP) is an algorithm for computing error gradients that bridges the gap between the Backpropagation (BP) algorithm and the Contrastive Hebbian Learning (CHL) algorithm used to train energy-based models (such as Boltzmann Machines and Hopfield networks).
EP is similar to CHL in that the learning rule to adjust the weights is local and Hebbian.
EP is also similar to BP in that it involves the propagation of an error signal backwards in the layers of the network.
These features make EP not only a model of interest for neuroscience, but also for the development of highly energy efficient learning-capable hardware (neuromorphic hardware).

Our recent NeurIPS (2019) paper:
* [Updates of Equilibrium Prop Match Gradients of Backprop Through Time in an RNN with Static Input](https://papers.nips.cc/paper/8930-updates-of-equilibrium-prop-match-gradients-of-backprop-through-time-in-an-rnn-with-static-input.pdf)
* [3-minute summary video](https://www.youtube.com/watch?v=Xb5sM0NRy_0&t=117s)

This paper introduces a discrete-time formulation of EP with simplified notations, closer to those used in the deep learning litterature.
It also establishes an equivalence of EP and BPTT in an RNN with static input and it introduces a convolutional RNN model trainable with EP.

Links to other papers:

* [Equivalence of Equilibrium Propagation and Recurrent Backpropagation](https://arxiv.org/abs/1711.08416)

* [Generalization of Equilibrium Propagation to Vector Field Dynamics](https://arxiv.org/abs/1808.04873)

* [Training a Spiking Neural Network with Equilibrium Propagation](http://proceedings.mlr.press/v89/o-connor19a/o-connor19a.pdf)

* [Initialized Equilibrium Propagation for Backprop-Free Training](https://openreview.net/pdf?id=B1GMDsR5tm)

The code of this repo is written in [Theano](https://github.com/Theano/Theano), the Deep Learning framework which was developed by [MILA](https://mila.umontreal.ca/en/).

[Click here](https://github.com/ernoult/updatesEPgradientsBPTT) for a more recent Keras implementation.

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
