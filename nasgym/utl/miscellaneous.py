"""Diverse methods for the environment management."""

import hashlib
import numpy as np
import datetime
import time


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
    # string = b("{phrase}".format(string))
    return hashlib.md5(string.encode()).hexdigest()


def state_to_string(state):
    """Encode a state into a string."""
    str_res = ""
    for layer in state:
        str_res += "{a}-{b}-{c}-{d};".format(
            a=layer[0],
            b=layer[1],
            c=layer[2],
            d=layer[3]
        )
    return str_res


def get_current_timestamp():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')