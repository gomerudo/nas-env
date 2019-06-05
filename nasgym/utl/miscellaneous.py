"""Diverse methods for the environment management."""

import datetime
import hashlib
import time
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
    """Normalize a dataset."""
    return dataset/np.float32(baseline)


def compute_str_hash(string):
    """Compute the hash of an string."""
    return hashlib.md5(string.encode()).hexdigest()


def state_to_string(state):
    """Encode a state into a string."""
    str_res = ""
    for layer in state:
        str_res += "{a}-{b}-{c}-{d}-{e};".format(
            a=layer[0],
            b=layer[1],
            c=layer[2],
            d=layer[3],
            e=layer[4],
        )
    return str_res


def get_current_layer(state):
    """Encode a state into a string."""
    return state[state.shape[0] - 1][0]


def get_current_timestamp():
    """Obtain the current system's timestamp."""
    current_time = time.time()
    return datetime.datetime.fromtimestamp(current_time).strftime(
        '%Y-%m-%d %H:%M:%S'
    )
