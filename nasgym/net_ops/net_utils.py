"""Utility net operations."""

import os
import numpy as np
import tensorflow as tf


def compute_network_density(graph, collection_name):
    """Compute the Density of a TensorFlow Neural Network."""
    graph = tf.get_default_graph()
    graph_def = graph.as_graph_def()

    nodes_counter = 0
    edges_counter = 0

    for node in graph_def.node:
        if node.name.startswith("{pre}/".format(pre=collection_name)):
            nodes_counter += 1
            edges_counter += len(node.input)

    # Note that we do not check for zero-division: on purpose to force failure.
    return edges_counter/nodes_counter


def compute_network_flops(graph, collection_name, logdir="workspace"):
    """Compute the Density of a TensorFlow Neural Network."""
    # Prepare the logdir
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    # Build the options
    opts = tf.profiler.ProfileOptionBuilder(
        tf.profiler.ProfileOptionBuilder.float_operation()
    ).with_node_names(
        start_name_regexes=["{name}.*".format(name=collection_name)]
    ).with_file_output(
        outfile="{dir}/flops.log".format(dir=logdir)
    ).build()

    # Get the flops object
    flops = tf.profiler.profile(
        graph,
        options=opts
    )

    # pylint: disable=no-member
    return flops.total_float_ops


def sort_sequence(sequence, as_list=True):
    """Sort the elements in the sequence, by layer_index."""
    if isinstance(sequence, np.ndarray):
        narray = sequence
    else:
        narray = np.array(sequence)

    narray = narray[narray[:, 0].argsort(kind='mergesort')]

    if as_list:
        return narray.tolist()
    else:
        return narray
