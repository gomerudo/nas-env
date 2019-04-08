"""Test the creation of the network."""

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


WORKSPACE_DIR = "./workspace"
GRAPHS_DIR = "{workspace}/graph".format(workspace=WORKSPACE_DIR)


class TestDefaultNASTrainer(unittest.TestCase):
    """Test the parsing of Neural Structure Code (NSC) to Network with TF."""

    def test_train(self):
        """Test the Default Training procedure."""
        tf.reset_default_graph()

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

        # Load training and eval data
        (train_data, train_labels), _ = tf.keras.datasets.mnist.load_data()

        # Fix the dataset
        train_data = normalize_dataset(dataset=train_data, baseline=255)
        train_labels = train_labels.astype(np.int32)

        nas_trainer = DefaultNASTrainer(
            network=net_nsc,
            input_shape=infer_data_shape(train_data),
            n_classes=infer_n_classes(train_labels),
            batch_size=256,
            log_path="./workspace/trainer_test",
            variable_scope="cnn"
        )
        print([n.name for n in tf.get_default_graph().as_graph_def().node])

        nas_trainer.train(
            train_data=train_data,
            train_labels=train_labels,
            train_input_fn="default"
        )

    def test_evaluate(self):
        """Test the Default Training procedure."""
        tf.reset_default_graph()

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

        # Load training and eval data
        (train_data, train_labels), (eval_data, eval_labels) = \
            tf.keras.datasets.mnist.load_data()

        # Fix the dataset
        train_data = normalize_dataset(dataset=train_data, baseline=255)
        train_labels = train_labels.astype(np.int32)

        nas_trainer = DefaultNASTrainer(
            network=net_nsc,
            input_shape=infer_data_shape(train_data),
            n_classes=infer_n_classes(train_labels),
            batch_size=256,
            log_path="./workspace/trainer_test"
        )

        nas_trainer.train(
            train_data=train_data,
            train_labels=train_labels,
            train_input_fn="default"
        )

        eval_data = normalize_dataset(dataset=eval_data, baseline=255)
        eval_labels = eval_labels.astype(np.int32)

        res = nas_trainer.evaluate(
            eval_data=eval_data,
            eval_labels=eval_labels,
            eval_input_fn="default"
        )

        print(res)


class TestFunctions(unittest.TestCase):
    """More."""

    def test_compute_densitity_network(self):
        """Test."""
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

        self.assertEqual(
            compute_network_density(tf.get_default_graph(), "cnn"),
            1.281767955801105
        )

    def test_compute_flops(self):
        """Test."""
        tf.reset_default_graph()

        with tf.variable_scope(name_or_scope="tnn"):
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

        print(compute_network_flops(tf.get_default_graph(), "cnn"))

        self.assertEqual(
            compute_network_flops(tf.get_default_graph(), "cnn"),
            75146
        )


class TestEarlyStopNASTrainer(unittest.TestCase):
    """Test the parsing of Neural Structure Code (NSC) to Network with TF."""

    def test_train(self):
        """Test the Default Training procedure."""
        tf.reset_default_graph()

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

        # Load training and eval data
        (train_data, train_labels), _ = tf.keras.datasets.mnist.load_data()

        # Fix the dataset
        train_data = normalize_dataset(dataset=train_data, baseline=255)
        train_labels = train_labels.astype(np.int32)

        nas_trainer = EarlyStopNASTrainer(
            network=net_nsc,
            input_shape=infer_data_shape(train_data),
            n_classes=infer_n_classes(train_labels),
            batch_size=256,
            log_path="./workspace/trainer_test",
            mu=0.5,
            rho=0.5,
            variable_scope="cnn"
        )

        nas_trainer.train(
            train_data=train_data,
            train_labels=train_labels,
            train_input_fn="default"
        )

    def test_evaluate(self):
        """Test the Default Training procedure."""
        tf.reset_default_graph()

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

        # Load training and eval data
        (train_data, train_labels), (eval_data, eval_labels) = \
            tf.keras.datasets.mnist.load_data()

        # Fix the dataset
        train_data = normalize_dataset(dataset=train_data, baseline=255)
        train_labels = train_labels.astype(np.int32)

        nas_trainer = EarlyStopNASTrainer(
            network=net_nsc,
            input_shape=infer_data_shape(train_data),
            n_classes=infer_n_classes(train_labels),
            batch_size=256,
            log_path="./workspace/trainer_test",
            mu=0.5,
            rho=0.5,
            variable_scope="cnn"
        )

        nas_trainer.train(
            train_data=train_data,
            train_labels=train_labels,
            train_input_fn="default"
        )

        eval_data = normalize_dataset(dataset=eval_data, baseline=255)
        eval_labels = eval_labels.astype(np.int32)

        res = nas_trainer.evaluate(
            eval_data=eval_data,
            eval_labels=eval_labels,
            eval_input_fn="default"
        )

        print(res)
        print(nas_trainer.density)
        print(nas_trainer.weighted_log_density)
        print(nas_trainer.flops)
        print(nas_trainer.weighted_log_flops)
