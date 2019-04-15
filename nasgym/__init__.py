"""Register the different NAS environments we makea available by default."""

import tensorflow as tf
from gym.envs.registration import register
from nasgym.dataset_handlers.default_handler import DefaultDatasetHandler


def register_nas_mnist():
    """Register NAS environment with default MNIST dataset."""
    (train_data, train_labels), (eval_data, eval_labels) = \
        tf.keras.datasets.mnist.load_data()

    handler = DefaultDatasetHandler(
        train_data, train_labels, eval_data, eval_labels, "mnist"
    )

    config_file = "nas-env/resources/nasenv.yml"

    register(
        id='NAS_mnist-v1',
        entry_point='nasgym.envs:DefaultNASEnv',
        kwargs={
            'config_file': config_file,
            'dataset_handler': handler
        }
    )


def register_nas_cifar10():
    """Register NAS environment with default CIFAR-10 dataset."""
    (train_data, train_labels), (eval_data, eval_labels) = \
        tf.keras.datasets.cifar10.load_data()

    handler = DefaultDatasetHandler(
        train_data, train_labels, eval_data, eval_labels, "cifar10"
    )

    config_file = "nas-env/resources/nasenv.yml"

    register(
        id='NAS_cifar10-v1',
        entry_point='nasgym.envs:DefaultNASEnv',
        kwargs={
            'config_file': config_file,
            'dataset_handler': handler
        }
    )


register_nas_mnist()
register_nas_mnist()