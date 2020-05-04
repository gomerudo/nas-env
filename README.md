# NASGym

A proof-of-concept OpenAI Gym environment for Neural Architecture Search (NAS).

**Under development:** The code and documentation of this repository may contain minor bugs.

## Overview

This is a python package developed for the research project [Learning to reinforcement learn for Neural Architecture Search](https://arxiv.org/abs/1911.03769).

The environment is fully compatible with the OpenAI baselines. It implements the RL steps for NAS, using the Neural Structure Code (NSC) of [BlockQNN: Efficient Block-wise Neural Network Architecture Generation](https://arxiv.org/abs/1808.05584) to encode the networks and make architectural changes. Under this setting, a Neural Network (i.e. the state for the reinforcement learning agent) is modeled as a list of NSCs, an action is the addition of a layer to the network, and the reward is the accuracy after the early-stop training. The datasets considered so far are the CIFAR-10 dataset (default) and the meta-dataset (it has to be downloaded as specified in [the original repository](https://github.com/gomerudo/meta-dataset)).

## Highlights

The code structure allows you to:

- Set the main training variables in a config.ini file (see the [resources](resources/) directory).
- Easily change the layers to consider and their hyperparameters in a yml file (see the [resources](resources/) directory).
- Create customized TensorFlow estimators to plug-in your performance estimation strategy.
- Store a local database of the experiments (CSV file by default, with an option to create your own DB interfaces).
- Create your own dataset handlers to input the TensorFlow estimators.
  
## Installation

The recommended way to install the package is in editable mode:

```
cd ${GIT_STORAGE}/nas-env
git checkout develop
pip install -e .
```
## Example

To see an example of how to use this environment after installation, check the scripts in the [nas-dmlr repo](https://github.com/gomerudo/nas-dmrl).
## Future plans

- Architectural changes to the source code to ease plugging-in the different NAS elements, i.e. the performance estimation strategy and the search space.
- Generate documentation
- Publish the package in PiPy.

## Contributing

So far I am the only contributor to this project, but I would like to improve the design to allow for even easier integration of the different NAS elements. Please feel free to join me in this quest!
