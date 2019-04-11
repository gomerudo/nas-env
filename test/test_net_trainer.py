"""Expose tests to verify the correct training of network's encoding.

In the NAS environment, the networks are represented in Neural Structure Code
(NSC). The nasgym/net_ops/net_trainer.py module is in charge of training these
networks on a given dataset, by first building the network with the functions
exposed in nasgym/net_ops/net_builder.py and then adding TensorFlow nodes to
the network to perform the training and evaluation.

Here, three test classes are included:

- TestFunctions: Test the functions in the `net_trainer` module, which do
        not belong to any class. Example: computation of FLOPs and
        Density of a NN.

- TestDefaultNASTrainer: In the `net_trainer` module different Trainers can be
        available. Here, we test the `DefaultNASTrainer`, which performs a
        typicall train procedure with cross-entropy minimization.

- TestEarlyStopNASTrainer: Another trainer available is the
        `EearlyStopNASTrainer`, which implements a typicall train procedure and
        evaluation, but additionally computes the FLOPs and Density of the
        Neural Network's graph. This is following the work described in
        https://arxiv.org/pdf/1808.05584.pdf.
"""

import os
import shutil
import unittest
import numpy as np
import tensorflow as tf
from nasgym.net_ops.net_trainer import compute_network_density
from nasgym.net_ops.net_trainer import compute_network_flops
from nasgym.net_ops.net_builder import sequence_to_net
from nasgym.net_ops.net_trainer import DefaultNASTrainer
from nasgym.net_ops.net_trainer import EarlyStopNASTrainer
from nasgym.utl.miscellaneous import infer_data_shape
from nasgym.utl.miscellaneous import infer_n_classes
from nasgym.utl.miscellaneous import normalize_dataset

# Set this always on top
tf.logging.set_verbosity(tf.logging.INFO)


class TestFunctions(unittest.TestCase):
    """Verify that the functions exposed in the module work correctly."""

    def setUp(self):
        """Set up of variables used in this test class."""
        # The short NSC used in this example
        self.net_nsc = [
            (1, 4, 0, 0, 0),  # Layer 1: Identity(input)
            (2, 1, 3, 1, 0),  # Layer 2: Convolution(Layer1)
            (3, 1, 3, 2, 0),  # Layer 3: Convolution(Layer2)
            (4, 5, 0, 1, 3),  # Layer 4: Convolution(Layer1)
            (5, 7, 0, 0, 0),  # Layer 5: Convolution(Layer4)
        ]

    def test_compute_densitity_network(self):
        """Verify density computation of the network defined in setUp().

        The test uses a pre-computed density over the network.

        We do not cover multiple cnn's coexisting in the graph, because
        TensorFlow adds some strange nodes if multiple grpahs are present.
        """
        tf.reset_default_graph()

        # Build an identical network, with a different scope.
        with tf.variable_scope(name_or_scope="cnn"):
            # Declare input
            input_placeholder = tf.placeholder(
                tf.float32,
                shape=(None, 28, 28, 1)
            )

            # Do the parsing
            sequence_to_net(self.net_nsc, input_placeholder)

        # Computed beforehand
        target_density = 1.281767955801105
        # Assert the value
        self.assertEqual(
            compute_network_density(tf.get_default_graph(), "cnn"),
            target_density
        )

    def test_compute_flops(self):
        """Verify the FLOPs computation of the network defined in setUp().

        The test uses a pre-computed FLOPs value over the network.

        This test case handles the case where more than 1 graph scopes are
        co-existing in the TensorFlow workspace, so that we verify that the
        FLOPs computation is not getting affected by those operations. This is
        needed because the computation of FLOPs depends on the correct
        filtering of the scope using regex.
        """
        tf.reset_default_graph()

        # Build a first neural network with one scope.
        with tf.variable_scope(name_or_scope="tnn"):
            # Declare input
            input_placeholder = tf.placeholder(
                tf.float32,
                shape=(None, 28, 28, 1)
            )

            # Do the parsing
            sequence_to_net(self.net_nsc, input_placeholder)

        # Build a second network, identical to the first one but in a different
        # scope.
        with tf.variable_scope(name_or_scope="cnn"):
            # Declare input
            input_placeholder = tf.placeholder(
                tf.float32,
                shape=(None, 28, 28, 1)
            )

            # Do the parsing
            sequence_to_net(self.net_nsc, input_placeholder)

        # Value computed beforehad
        target_flops = 75146
        # Make the assert operation.
        self.assertEqual(
            compute_network_flops(tf.get_default_graph(), "cnn"),
            target_flops
        )


class TestDefaultNASTrainer(unittest.TestCase):
    """Test that training is executed correcty with DefaultNASTrainer.

    The training procedure of a NN is stochastic, hence, we focus on the
    correct execution in terms of programming runtime (i.e. no exceptions) and
    the correct creation of TensorFlow derived objects (DOs).
    """

    def setUp(self):
        """Set up of variables used in this test class."""
        # The short NSC used in this example
        self.net_nsc = [
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

        # Load training and eval data
        (train_data, train_labels), (eval_data, eval_labels) = \
            tf.keras.datasets.mnist.load_data()

        # Fix the dataset
        self.train_data = normalize_dataset(dataset=train_data, baseline=255)
        self.train_labels = train_labels.astype(np.int32)

        self.eval_data = normalize_dataset(dataset=eval_data, baseline=255)
        self.eval_labels = eval_labels.astype(np.int32)

        # The batch size
        self.batch_size = 256

        # Workspace directory
        workspace_dir = "./workspace"
        self.training_dir = "{workspace}/trainer_test".format(
            workspace=workspace_dir
        )

    def test_train(self):
        """Test the Default Training procedure."""
        tf.reset_default_graph()
        if os.path.isdir(self.training_dir):
            shutil.rmtree(self.training_dir)

        nas_trainer = DefaultNASTrainer(
            encoded_network=self.net_nsc,
            input_shape=infer_data_shape(self.train_data),
            n_classes=infer_n_classes(self.train_labels),
            batch_size=self.batch_size,
            log_path=self.training_dir,
            variable_scope="cnn"
        )

        nas_trainer.train(
            train_data=self.train_data,
            train_labels=self.train_labels,
            train_input_fn="default"
        )

        self.assertTrue(os.path.isdir(self.training_dir))

    def test_evaluate(self):
        """Test the Default Training procedure."""
        tf.reset_default_graph()
        if os.path.isdir(self.training_dir):
            shutil.rmtree(self.training_dir)

        nas_trainer = DefaultNASTrainer(
            encoded_network=self.net_nsc,
            input_shape=infer_data_shape(self.train_data),
            n_classes=infer_n_classes(self.train_labels),
            batch_size=self.batch_size,
            log_path=self.training_dir,
            variable_scope="cnn"
        )

        nas_trainer.train(
            train_data=self.train_data,
            train_labels=self.train_labels,
            train_input_fn="default"
        )

        res = nas_trainer.evaluate(
            eval_data=self.eval_data,
            eval_labels=self.eval_labels,
            eval_input_fn="default"
        )

        self.assertTrue(os.path.isdir(self.training_dir))
        self.assertTrue("accuracy" in list(res.keys()))


class TestEarlyStopNASTrainer(unittest.TestCase):
    """Test the parsing of Neural Structure Code (NSC) to Network with TF."""

    def setUp(self):
        """Set up of variables used in this test class."""
        # The short NSC used in this example
        self.net_nsc = [
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

        # Load training and eval data
        (train_data, train_labels), (eval_data, eval_labels) = \
            tf.keras.datasets.mnist.load_data()

        # Fix the dataset
        self.train_data = normalize_dataset(dataset=train_data, baseline=255)
        self.train_labels = train_labels.astype(np.int32)

        self.eval_data = normalize_dataset(dataset=eval_data, baseline=255)
        self.eval_labels = eval_labels.astype(np.int32)

        # The batch size
        self.batch_size = 256

        # Workspace directory
        workspace_dir = "./workspace"
        self.training_dir = "{workspace}/trainer_test".format(
            workspace=workspace_dir
        )

    def test_train(self):
        """Test the Default Training procedure."""
        tf.reset_default_graph()
        if os.path.isdir(self.training_dir):
            shutil.rmtree(self.training_dir)

        nas_trainer = EarlyStopNASTrainer(
            encoded_network=self.net_nsc,
            input_shape=infer_data_shape(self.train_data),
            n_classes=infer_n_classes(self.train_labels),
            batch_size=self.batch_size,
            log_path=self.training_dir,
            mu=0.5,
            rho=0.5,
            variable_scope="cnn"
        )

        nas_trainer.train(
            train_data=self.train_data,
            train_labels=self.train_labels,
            train_input_fn="default"
        )

        self.assertTrue(os.path.isdir(self.training_dir))

    def test_evaluate(self):
        """Test the Default Training procedure."""
        tf.reset_default_graph()
        if os.path.isdir(self.training_dir):
            shutil.rmtree(self.training_dir)

        nas_trainer = EarlyStopNASTrainer(
            encoded_network=self.net_nsc,
            input_shape=infer_data_shape(self.train_data),
            n_classes=infer_n_classes(self.train_labels),
            batch_size=self.batch_size,
            log_path=self.training_dir,
            mu=0.5,
            rho=0.5,
            variable_scope="cnn"
        )

        nas_trainer.train(
            train_data=self.train_data,
            train_labels=self.train_labels,
            train_input_fn="default"
        )

        res = nas_trainer.evaluate(
            eval_data=self.eval_data,
            eval_labels=self.eval_labels,
            eval_input_fn="default"
        )

        self.assertTrue(os.path.isdir(self.training_dir))
        self.assertTrue("accuracy" in list(res.keys()))

        self.assertTrue(nas_trainer.density is not None)
        self.assertTrue(nas_trainer.density != 0.)

        self.assertTrue(nas_trainer.weighted_log_density is not None)
        self.assertTrue(nas_trainer.weighted_log_density != 0.)

        self.assertTrue(nas_trainer.flops is not None)
        self.assertTrue(nas_trainer.flops != 0.)

        self.assertTrue(nas_trainer.weighted_log_flops is not None)
        self.assertTrue(nas_trainer.weighted_log_flops != 0.)
