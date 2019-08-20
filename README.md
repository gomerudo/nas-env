# NASGym

A simple OpenAI Gym environment for Neural Architecture Search (NAS).

**Under development:** The code and documentation of this repository may contain minor bugs.

## Overview

This is a python package developed for the *Learning to reinforcement learn for Neural Architecture Search* MSc thesis at the Eindhoven University of Technology (TU/e).

The environment is fully-compatible with the OpenAI baselines and exposes a NAS environment following the Neural Structure Code of [BlockQNN: Efficient Block-wise Neural Network Architecture Generation](https://arxiv.org/abs/1808.05584). Under this setting, a Neural Network (i.e. the state for the reinforcement learning agent) is modeled as a list of NSCs, an action is the addition of a layer to the network, and the reward is the accuracy after the early-stop training. The datasets considered so far are the CIFAR-10 dataset (available by default) and the meta-dataset (has to be manually downloaded as specified in [this repository](https://github.com/gomerudo/meta-dataset)).

The logic implemented allows to customize the behaviour of the environment, although some coding is still needed.

## Installation

The recommend way to install the package is in editable mode:

```
cd ${GIT_STORAGE}/nas-env
git checkout develop
pip install -e .
```

## Future plans

- Architectural changes to the source code to allow to plug-in the different NAS elements, i.e. the performance estimation strategy and the search space.
- Generate documentation
- Publish the package in PiPy.

## Contributing

So far I am the only contributor to this project, but I would like to improve the design to allow for an even easier integration of different methodologies for the environment. In this way, different NAS elements can be plugged-in 