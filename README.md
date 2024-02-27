# Equilibrium Propagation

This repo contains the code of the original [equilibrium propagation](https://www.frontiersin.org/articles/10.3389/fncom.2017.00024/full) (EP) paper. The code is written in [Theano](https://github.com/Theano/Theano), the framework once developed by [Mila](https://mila.quebec/en/).

For more recent code:
* [This repo](https://github.com/rain-neuromorphics/energy-based-learning) contains code written in PyTorch.
* [This repo](https://github.com/Laborieux-Axel/holomorphic_eqprop) contains code for 'holomorphic equilibrium propagation', written in Jax.

Other repositories with code for EP: [1](https://github.com/Laborieux-Axel/Equilibrium-Propagation), [2](https://github.com/smonsays/equilibrium-propagation), [3](https://github.com/ernoult/updatesEPgradientsBPTT).

## Getting started
* Download the code from GitHub:
```bash
git clone https://github.com/bscellier/Towards-a-Biologically-Plausible-Backprop
cd Towards-a-Biologically-Plausible-Backprop
```
* To train a Hopfield network (with 1 hidden layer by default) with Eqprop, run the command:
``` bash
THEANO_FLAGS="floatX=float32, gcc.cxxflags='-march=core2'" python train_model.py
```
* Once the network is trained, use the GUI by running the command:
``` bash
THEANO_FLAGS="floatX=float32, gcc.cxxflags='-march=core2'" python gui.py net1
```