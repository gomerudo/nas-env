"""Test the creation of the network."""

import os
import unittest
import numpy as np
from gym import spaces
from nasgym.envs.default_nas_env import DefaultNASEnvParser
from nasgym.envs.default_nas_env import DefaultNASEnv

WORKSPACE_DIR = "./workspace"
GRAPHS_DIR = "{workspace}/graph".format(workspace=WORKSPACE_DIR)


class TestDefaultNASParser(unittest.TestCase):
    """Test the parsing of Neural Structure Code (NSC) to Network with TF."""

    def test_observation_space(self):
        """Test the type of the observation space."""
        current_dir = os.getcwd()
        nas_yml_path = "{root_dir}/{name}".format(
            root_dir=current_dir,
            name="nasenv.yml"
        )
        dnase_parser = DefaultNASEnvParser(nas_yml_path)

        # Assert the type
        self.assertTrue(isinstance(dnase_parser.observation_space, spaces.Box))

        # Assert the size
        expected_shape = (10, 5)
        self.assertEqual(dnase_parser.observation_space.shape, expected_shape)

    def test_action_space(self):
        """Test the type of the action space."""
        current_dir = os.getcwd()
        nas_yml_path = "{root_dir}/{name}".format(
            root_dir=current_dir,
            name="nasenv.yml"
        )
        dnase_parser = DefaultNASEnvParser(nas_yml_path)

        # Assert the type
        self.assertTrue(isinstance(dnase_parser.action_space, spaces.Discrete))

        # Assert the size
        expected_dim = 341
        self.assertEqual(dnase_parser.action_space.n, expected_dim)


class TestDefaultNASEnv(unittest.TestCase):
    """Test the behavior of the DefaultNASEnv."""

    def test_environment_creation(self):
        """Test the creation of the environment."""
        # Creation of the environment
        assigned_max_steps = 10
        assigned_max_layers = 10

        nasenv = DefaultNASEnv(
            config_file="./nasenv.yml",
            max_steps=assigned_max_steps,
            max_layers=assigned_max_layers,
            dataset='meta-dataset',
            is_learning=True
        )

        # Assert the basic properties are set correctly
        self.assertFalse(np.count_nonzero(nasenv.state))
        self.assertTrue(nasenv.max_steps == assigned_max_steps)
        self.assertTrue(nasenv.max_layers == assigned_max_layers)

        self.assertTrue(isinstance(nasenv.observation_space, spaces.Box))
        self.assertTrue(isinstance(nasenv.action_space, spaces.Discrete))
        self.assertTrue(isinstance(nasenv.actions_info, dict))

        # Assert the dimension of the action space: it should be 288 for
        # default configuration
        expected_dim = 341
        self.assertEqual(nasenv.action_space.n, expected_dim)

    def test_action_flow(self):
        """Test the creation of the environment."""
        # Creation of the environment
        nasenv = DefaultNASEnv(
            config_file="./nasenv.yml",
            max_steps=10,
            max_layers=10,
            dataset='meta-dataset',
            is_learning=True
        )

        set_actions = [0, 280, 2, 3, 4, 5, 6, 7, 8, 9]

        for action in set_actions:
            state, reward, done, info = nasenv.step(action)
            print("State:\n {s}".format(s=state))
            print("Reward: {s}".format(s=reward))
            print("Done: {s}".format(s=done))
            print("Info: {s}".format(s=info))
            if done:
                break
