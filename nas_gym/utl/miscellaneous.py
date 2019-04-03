"""Diverse methods for the environment management."""

import numpy as np


def is_valid_config_file(config_file):
    """Verify whether or not the configuration file is valid."""
    return config_file.lower().endswith('.yml')


def infer_data_shape(dataset):
    """From a dataset, infer the shape of each observation."""
    return tuple(list(dataset.shape)[1:])


def infer_n_classes(labels):
    """From a list of labels, infer the total number of classes."""
    return np.unique(labels).shape[0]


def normalize_dataset(dataset, baseline=255):
    """Normalize a dataset"""
    return dataset/np.float32(255)