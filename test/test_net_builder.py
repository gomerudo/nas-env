"""Test the building of networks from Neural Structure Code representations.

In the NAS environment, we support the Neural Structure Code (NSC) from
BlockQNN (see https://arxiv.org/abs/1808.05584). In the module
nasgym/net_ops/net_builder we have exposed functions to build Neural Networks
from this NSC. We use the default type of layers from the paper, and try to
build neural networks. However, testing the creation of such networks is not
feasible in this kind of automated tests, hence, we test some simple variables
that can indicate that the creation performed correctly.

By design, all the networks tested here are valid and are supposed to be built
correctly. This was manually verified with the help of TensorBoard. If these
tests break, that indicates a clear break in the building of the network. We
note hoewever that it is not feasible to write more robust tests that the ones
included here.
"""

import os
import shutil
import unittest
import tensorflow as tf
from nasgym.net_ops.net_builder import sequence_to_net
from nasgym.net_ops.net_utils import sort_sequence

# Set this always on top
tf.logging.set_verbosity(tf.logging.INFO)


# TODO: test manually that the graph_def has the elements we specified
# TODO: test manually the final dimension of the tensor.
class TestParsing(unittest.TestCase):
    """Test the parsing of Neural Structure Code (NSC) to Network with TF."""

    def setUp(self):
        """Set up of variables used in this test class."""
        workspace_dir = "./workspace"
        self.graphs_dir = "{workspace}/graph".format(workspace=workspace_dir)

    def test_sequence_to_net_long(self):
        """Verify that the long NSC sequence is build correctly."""
        tf.reset_default_graph()
        custom_graph_dir = self.graphs_dir + "/test01"
        if os.path.isdir(custom_graph_dir):
            shutil.rmtree(custom_graph_dir)

        with tf.variable_scope(name_or_scope="cnn"):
            net_nsc = [
                (1, 4, 0, 0, 0),  # Layer 1: Identity(input)
                (2, 1, 1, 1, 0),  # Layer 2: Convolution(Layer1)
                (3, 1, 3, 2, 0),  # Layer 3: Convolution(Layer2)
                (4, 1, 1, 1, 0),  # Layer 4: Convolution(Layer1)
                (5, 1, 5, 4, 0),  # Layer 5: Convolution(Layer4)
                (6, 6, 0, 3, 5),  # Layer 6: Concat(Layer3, Layer5)
                (7, 2, 3, 1, 0),  # Layer 7: MaxPooling(Layer1)
                (8, 1, 1, 7, 0),  # Layer 8: Convolution(Layer7)
                (9, 6, 0, 6, 8),  # Layer 9: Concat(Layer6, Layer8)
                (10, 7, 0, 0, 0),  # Layer 10: Terminal
            ]

            # Declare input
            input_placeholder = tf.placeholder(
                tf.float32,
                shape=(None, 28, 28, 1)
            )

            # Do the parsing
            sequence_to_net(net_nsc, input_placeholder)

            # Save the graph
            file_writer = tf.summary.FileWriter(
                custom_graph_dir,
                tf.get_default_graph()
            )
            file_writer.close()

        self.assertTrue(os.path.isdir(custom_graph_dir))

    def test_sequence_to_net_concat_nonused(self):
        """Verify that the concatenation of non-used layers is correct."""
        tf.reset_default_graph()
        custom_graph_dir = self.graphs_dir + "/test02"
        if os.path.isdir(custom_graph_dir):
            shutil.rmtree(custom_graph_dir)

        with tf.variable_scope(name_or_scope="cnn"):
            net_nsc = [
                (1, 4, 0, 0, 0),  # Layer 1: Identity(input)
                (2, 1, 1, 1, 0),  # Layer 2: Convolution(Layer1)
                (3, 1, 3, 2, 0),  # Layer 3: Convolution(Layer2)
                (4, 1, 1, 1, 0),  # Layer 4: Convolution(Layer1)
                (5, 1, 5, 4, 0),  # Layer 5: Convolution(Layer4)
                (6, 2, 3, 1, 0),  # Layer 7: MaxPooling(Layer1)
                (7, 1, 1, 6, 0),  # Layer 8: Convolution(Layer7)
                (8, 7, 0, 0, 0),  # Layer 10: Terminal
            ]

            # Declare input
            input_placeholder = tf.placeholder(
                tf.float32,
                shape=(None, 28, 28, 1)
            )

            # Do the parsing
            sequence_to_net(net_nsc, input_placeholder)

            # Save the graph
            file_writer = tf.summary.FileWriter(
                custom_graph_dir,
                tf.get_default_graph()
            )
            file_writer.close()

            # graph = tf.get_default_graph()
        self.assertTrue(os.path.isdir(custom_graph_dir))

    def test_sequence_to_net_short(self):
        """Verify that the short NSC sequence is build correctly."""
        tf.reset_default_graph()
        custom_graph_dir = self.graphs_dir + "/test03"
        if os.path.isdir(custom_graph_dir):
            shutil.rmtree(custom_graph_dir)

        with tf.variable_scope(name_or_scope="cnn"):
            net_nsc = [
                (1, 4, 0, 0, 0),  # Layer 1: Identity(input)
                (2, 1, 3, 1, 0),  # Layer 2: Convolution(Layer1)
                (3, 1, 3, 2, 0),  # Layer 3: Convolution(Layer2)
                (4, 5, 0, 1, 3),  # Layer 4: Convolution(Layer1)
                (5, 7, 0, 0, 0),  # Layer 5: Convolution(Layer4)
            ]

            # Declare input
            input_placeholder = tf.placeholder(
                tf.float32,
                shape=(None, 28, 28, 1)
            )

            # Do the parsing
            sequence_to_net(net_nsc, input_placeholder)

            # Save the graph
            file_writer = tf.summary.FileWriter(
                custom_graph_dir,
                tf.get_default_graph()
            )
            file_writer.close()

        self.assertTrue(os.path.isdir(custom_graph_dir))

    def test_sort_sequence(self):
        """Verify that the sorting of a NSC sequence is correct."""
        original_sequence = [
            [8, 8, 0, 8, 0],
            [1, 1, 0, 1, 0],
            [6, 6, 0, 6, 0],
            [10, 10, 0, 10, 0],
            [4, 4, 0, 4, 0],
            [3, 3, 0, 3, 0],
            [2, 2, 0, 2, 0],
            [7, 7, 0, 7, 0],
            [9, 9, 0, 9, 0],
            [5, 5, 0, 5, 0],
        ]

        expected_sequence = [
            [1, 1, 0, 1, 0],
            [2, 2, 0, 2, 0],
            [3, 3, 0, 3, 0],
            [4, 4, 0, 4, 0],
            [5, 5, 0, 5, 0],
            [6, 6, 0, 6, 0],
            [7, 7, 0, 7, 0],
            [8, 8, 0, 8, 0],
            [9, 9, 0, 9, 0],
            [10, 10, 0, 10, 0],
        ]

        # Make conversion
        res = sort_sequence(original_sequence)

        # Assert the result
        self.assertTrue(res == expected_sequence)

    def test_sequence_to_net_addlayer(self):
        """Verify that the safe_add() method works correctly."""
        tf.reset_default_graph()
        custom_graph_dir = self.graphs_dir + "/test04"
        if os.path.isdir(custom_graph_dir):
            shutil.rmtree(custom_graph_dir)

        with tf.variable_scope(name_or_scope="cnn"):
            net_nsc = [
                (1, 4, 0, 0, 0),  # Layer 1: Identity(input)
                (2, 1, 1, 1, 0),  # Layer 2: Convolution(Layer1)
                (3, 1, 3, 2, 0),  # Layer 3: Convolution(Layer2)
                (4, 1, 1, 1, 0),  # Layer 4: Convolution(Layer1)
                (5, 1, 5, 4, 0),  # Layer 5: Convolution(Layer4)
                (6, 5, 0, 3, 5),  # Layer 6: Add(Layer3, Layer5)
                (7, 2, 3, 1, 0),  # Layer 7: MaxPooling(Layer1)
                (8, 1, 1, 7, 0),  # Layer 8: Convolution(Layer7)
                (9, 5, 0, 6, 8),  # Layer 9: Add(Layer6, Layer8)
                (10, 7, 0, 0, 0),  # Layer 10: Terminal
            ]

            # Declare input
            input_placeholder = tf.placeholder(
                tf.float32,
                shape=(None, 28, 28, 1)
            )

            # Do the parsing
            sequence_to_net(net_nsc, input_placeholder)

            # Save the graph
            file_writer = tf.summary.FileWriter(
                custom_graph_dir,
                tf.get_default_graph()
            )
            file_writer.close()

        self.assertTrue(os.path.isdir(custom_graph_dir))
