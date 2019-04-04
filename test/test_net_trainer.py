"""Test the creation of the network."""

import unittest
import numpy as np
import tensorflow as tf
from nasgym.net_ops.net_trainer import DefaultNASTrainer
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
                log_path="./workspace/trainer_test"
            )

            nas_trainer.train(
                train_data=train_data,
                train_labels=train_labels,
                train_input_fn="default"
            )

    def test_evaluate(self):
        """Test the Default Training procedure."""
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
