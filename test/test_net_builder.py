"""Test the creation of the network"""

import unittest
import tensorflow as tf
from nas_gym.net_ops.net_builder import sequence_to_net

WORKSPACE_DIR = "./workspace"
GRAPHS_DIR = "{workspace}/graph".format(workspace=WORKSPACE_DIR)


class TestParsing(unittest.TestCase):
    """Test the parsing of Neural Structure Code (NSC) to Network with TF."""

    def test_sequence_to_net_wplaceholder(self):
        """Test intended to check that the graph is built correctly."""
        tf.reset_default_graph()

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
                GRAPHS_DIR,
                tf.get_default_graph()
            )
            file_writer.close()

    def test_sequence_to_net_wplaceholder_b(self):
        """Test intended to check that the graph is built correctly."""
        tf.reset_default_graph()

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
                GRAPHS_DIR,
                tf.get_default_graph()
            )
            file_writer.close()

