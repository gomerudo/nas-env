"""Test simple workflows."""

import unittest
import numpy as np
from nasgym.envs.state_encoders import DefaultEncoder


class TestDefaultEncoder(unittest.TestCase):
    """Encoding tests."""

    def test_encoding(self):
        """Execute a simple training procedure."""
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

        encoder = DefaultEncoder(
            n_types=7, max_layers=10, max_kernel=5, is_multi_branch=False
        )
        encoded_arch = encoder.encode(architecture)
        print(encoded_arch.shape)
        print(encoded_arch)
