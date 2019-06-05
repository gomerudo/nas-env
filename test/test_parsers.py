"""Test the creation of the network."""

import os
import unittest
import numpy as np
import tensorflow as tf
from gym import spaces
from nasgym.envs.envspecs_parsers import ChainedEnvParser
from nasgym.dataset_handlers.default_handler import DefaultDatasetHandler

NAS_YML_FILE = "{root_dir}/{name}".format(
    root_dir=os.getcwd(),
    name="resources/nasenv.yml"
)


class TestDefaultNASParser(unittest.TestCase):
    """Test the parsing of Neural Structure Code (NSC) to Network with TF."""

    def test_observation_space(self):
        """Test the type of the observation space."""
        dnase_parser = ChainedEnvParser(NAS_YML_FILE)

        # Assert the type
        self.assertTrue(isinstance(dnase_parser.observation_space, spaces.Box))

        # Assert the size
        expected_shape = (10, 5)
        self.assertEqual(dnase_parser.observation_space.shape, expected_shape)

    def test_action_space(self):
        """Test the type of the action space."""
        dnase_parser = ChainedEnvParser(NAS_YML_FILE)

        # Assert the type
        # self.assertTrue(isinstance(dnase_parser.action_space, spaces.Discrete))
        space, info = dnase_parser.action_space
        print(info)
        print(space)
        # Assert the size
        # x = dnase_parser.max_nlayers  # pylint: disable=invalid-name
        # expected_dim = x*x + 6*x + 1
        # self.assertEqual(dnase_parser.action_space.n, expected_dim)
