"""Test simple workflows."""

import os
import unittest
import numpy as np
from nasgym import nas_logger
from nasgym.envs.factories import DatasetHandlerFactory
from nasgym.envs.factories import TrainerFactory
from nasgym.utl.miscellaneous import compute_str_hash
from nasgym.utl.miscellaneous import state_to_string


class TestCustom(unittest.TestCase):
    """Various basic tests."""

    def setUp(self):
        """Set up of variables used in this test class."""
        self.yml_file = "{root_dir}/{name}".format(
            root_dir=os.getcwd(),
            name="resources/nasenv.yml"
        )

        self.workspace_dir = "./workspace"

    def test_training(self):
        """Execute a simple training procedure."""
        dataset_handler = DatasetHandlerFactory.get_handler(
            handler_type="meta-dataset"
        )

        architecture = np.array([
            [0, 0, 0, 0, 0],  # 1
            [0, 0, 0, 0, 0],  # 2
            [0, 0, 0, 0, 0],  # 3
            [0, 0, 0, 0, 0],  # 4
            [0, 0, 0, 0, 0],  # 5
            [0, 0, 0, 0, 0],  # 6
            [1, 1, 3, 0, 0],  # 7
            [2, 1, 5, 1, 0],  # 8
            [3, 1, 1, 2, 0],  # 9
            [4, 7, 0, 3, 0],  # 10
        ])

        hash_state = compute_str_hash(state_to_string(architecture))
        composed_id = "{d}-{h}".format(
            d=dataset_handler.current_dataset_name(), h=hash_state
        )

        nas_trainer = TrainerFactory.get_trainer(
            trainer_type="default",
            state=architecture.copy(),
            dataset_handler=dataset_handler,
            log_path="trainer-{h}".format(h=hash_state),
            variable_scope="cnn-{h}".format(h=hash_state)
        )

        def custom_train_input_fn():
            return dataset_handler.current_train_set()

        def custom_eval_input_fn():
            return dataset_handler.current_validation_set()

        train_input_fn = custom_train_input_fn
        eval_input_fn = custom_eval_input_fn

        nas_trainer.train(
            train_data=None,
            train_labels=None,
            train_input_fn=train_input_fn,
            n_epochs=12  # As specified by BlockQNN
        )

        nas_logger.debug("Evaluating architecture %s", composed_id)
        res = nas_trainer.evaluate(
            eval_data=None,
            eval_labels=None,
            eval_input_fn=eval_input_fn
        )
        nas_logger.debug(
            "Train-evaluation procedure finished for architecture %s",
            composed_id
        )

        accuracy = res['accuracy']
        print(accuracy)
