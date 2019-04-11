"""Classes and methods for Net building."""

import logging
import numpy as np
import tensorflow as tf
from nasgym.net_ops import LTYPE_ADD
from nasgym.net_ops import LTYPE_AVGPOOLING
from nasgym.net_ops import LTYPE_CONCAT
from nasgym.net_ops import LTYPE_CONVULUTION
from nasgym.net_ops import LTYPE_IDENTITY
from nasgym.net_ops import LTYPE_MAXPOOLING
from nasgym.net_ops import LTYPE_TERMINAL


def sequence_to_net(sequence, input_tensor):
    """Build a network with TensorFlow, given a sequence of NSC."""
    # We use this list to store the built layers. Remember that each time we
    # iterate, we find the predecesor of the current layer, hence, the inputs
    # are never a problem. If for some reason the layer has no input, we should
    # throw an exception and ignore the layer
    tf_layers = {
        0: input_tensor
    }
    current_layer = None

    # We use this list to see if all layers were used to construct the graph
    non_used_layers = list(set([layer[0] for layer in sequence]))

    for layer_encoding in sequence:
        layer_index = layer_encoding[0]
        layer_type = layer_encoding[1]
        layer_kernel_size = layer_encoding[2]
        layer_pred1 = layer_encoding[3]  # Predecesor 1
        layer_pred2 = layer_encoding[4]  # Predecesor 2

        if not layer_index:
            continue

        if layer_type == LTYPE_ADD:
            # i.e. if no predecesors at all
            if not layer_pred1 or not layer_pred2:
                logging.warning("No predecessors for layer %d" % layer_index)

            # TODO: check if exception is raised for non existant tf_layers[i]
            with tf.name_scope("L{i}_ADD".format(i=layer_index)):
                current_layer = safe_add(
                    tensor_a=tf_layers[layer_pred1],
                    tensor_b=tf_layers[layer_pred2],
                    name="SafeAdd"
                )

        if layer_type == LTYPE_AVGPOOLING:
            with tf.name_scope("L{i}_AVGPOOL".format(i=layer_index)):
                current_layer = tf.keras.layers.AveragePooling2D(
                    pool_size=layer_kernel_size,
                    name="AvgPooling"
                )(tf_layers[layer_pred1])

        if layer_type == LTYPE_CONCAT:
            # i.e. if no predecesors at all
            if not layer_pred1 or not layer_pred2:
                logging.warning("No predecessors for layer %i" % layer_index)

            with tf.name_scope("L{i}_CONCAT".format(i=layer_index)):
                current_layer = safe_concat(
                    tensor_a=tf_layers[layer_pred1],
                    tensor_b=tf_layers[layer_pred2],
                    name="SafeConcat"
                )

        if layer_type == LTYPE_CONVULUTION:
            with tf.name_scope("L{i}_PCC".format(i=layer_index)):
                batch_norm = tf.keras.layers.BatchNormalization(
                    name="BatchNorm"
                )(tf_layers[layer_pred1])

                relu_layer = tf.keras.layers.ReLU(
                    name="ReLU"
                )(batch_norm)
                # )(tf_layers[layer_pred1])

                conv_layer = tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=layer_kernel_size,
                    padding="same",
                    name="Conv"
                )(relu_layer)

                current_layer = conv_layer

        if layer_type == LTYPE_IDENTITY:
            with tf.name_scope("L{i}_IDENTITY".format(i=layer_index)):
                current_layer = tf.identity(
                    input=tf_layers[layer_pred1],
                    name="Identity"
                )

        if layer_type == LTYPE_MAXPOOLING:
            with tf.name_scope("L{i}_MAXPOOL".format(i=layer_index)):
                current_layer = tf.keras.layers.MaxPool2D(
                    pool_size=layer_kernel_size,
                    name="MaxPooling",
                    padding="same"
                )(tf_layers[layer_pred1])

        if layer_type == LTYPE_TERMINAL:
            # We remove the terminal layer because we already visited it.
            try:
                non_used_layers.remove(layer_index)
            except ValueError:
                pass

            # Force the end of the building process. We ignore any remaining
            # portion of the sequence.
            break

        # Add the current layer to the dictionary
        tf_layers[layer_index] = current_layer

        # Mark as used:
        #   Two different try-except to always remove both of the predecesors
        try:
            non_used_layers.remove(layer_pred1)
        except ValueError:
            pass

        try:
            non_used_layers.remove(layer_pred2)
        except ValueError:
            pass

    # Handle the non used layers: concatenate one by one.
    #   The last layer before terminate is never used, but if there are two or
    #   more, then we need to concatenate one by one.
    if non_used_layers:
        # print("Non used", non_used_layers)
        pivot = tf_layers[non_used_layers[0]]
        for idx in range(1, len(non_used_layers)):
            with tf.name_scope("END_CONCAT{i}".format(i=idx)):
                pivot = safe_concat(
                    tensor_a=pivot,
                    tensor_b=tf_layers[non_used_layers[idx]],
                    name="SafeConcat"
                )
        current_layer = pivot

    # current_layer is the last used layer
    return current_layer


def safe_concat(tensor_a, tensor_b, name):
    """Concatenate two tensors even if they have different shapes.

    The fix of the shapes is done with a zero-padding on both tensors.
    """
    fixed_b = fix_tensor_shape(
        tensor_target=tensor_b,
        tensor_reference=tensor_a,
        free_axis=1
    )
    fixed_a = fix_tensor_shape(
        tensor_target=tensor_a,
        tensor_reference=tensor_b,
        free_axis=1
    )

    concatenated = tf.keras.layers.concatenate(
        inputs=[
            fixed_a,
            fixed_b,
        ],
        name=name
    )

    return concatenated


def safe_add(tensor_a, tensor_b, name):
    """Concatenate two tensors even if they have different shapes.

    The fix of the shapes is done with a zero-padding on both tensors.
    """
    fixed_b = fix_tensor_shape(
        tensor_target=tensor_b,
        tensor_reference=tensor_a,
        free_axis=1
    )
    fixed_a = fix_tensor_shape(
        tensor_target=tensor_a,
        tensor_reference=tensor_b,
        free_axis=1
    )

    added = tf.keras.layers.add(
        inputs=[
            fixed_a,
            fixed_b,
        ],
        name=name
    )

    return added


def is_same_rank(tensor_a, tensor_b):
    """Verify whether the rank of two tensors is the same."""
    return tensor_a.get_shape().rank == tensor_b.get_shape().rank


def fix_tensor_shape(tensor_target, tensor_reference, free_axis=1, name="pad"):
    """Fix a tensor's shape with respect to a reference using padding."""
    ref_shape = tensor_reference.get_shape().dims
    target_shape = tensor_target.get_shape().dims
    target_rank = len(target_shape)
    ref_rank = len(ref_shape)

    if ref_rank != target_rank:
        raise ValueError("Tensors must have the same dimension.")

    free_axis -= 1  # Shift the axis to start with 0 for simplicity

    paddings_arg = []
    for it_axis in range(target_rank):
        # If the current axis is the free axis then pad nothing, i.e. (rank, 0)
        if it_axis == free_axis:
            paddings_arg.append([0, 0])
            continue

        # Compute the difference of the axes to know how many pads are needed
        if ref_shape[it_axis].value is None or \
                target_shape[it_axis].value is None:
            axes_diff = 0
        else:
            axes_diff = ref_shape[it_axis].value - target_shape[it_axis].value

        # If target axis has a higher order than reference then do not pad.
        if axes_diff < 0:
            axes_diff = 0

        # If everything is ok, we simply store the desired fix-value
        paddings_arg.append([axes_diff//2, axes_diff - axes_diff//2])

    # print("Padding with list", paddings_arg)

    padded_tensor = tf.pad(
        tensor=tensor_target,
        paddings=tf.constant(paddings_arg),
        mode="CONSTANT",
        constant_values=0,
        name=name + "padding"
    )

    return padded_tensor


def sort_sequence(sequence, as_list=True):
    """Sort the elements in the sequence, by layer_index."""
    narray = np.array(sequence)
    narray = narray[narray[:, 0].argsort(kind='mergesort')]

    if as_list:
        return narray.tolist()
    else:
        return narray
