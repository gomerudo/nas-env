"""Possible state encoders to use."""
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class DefaultEncoder:
    """Encode a given state from the NAS environment with the default logic."""

    def __init__(self, n_types=7, max_layers=10, max_kernel=5,
                 is_multi_branch=False):
        """Constructor."""
        self.max_layers = max_layers
        self.max_kernel = max_kernel
        self.is_multi_branch = is_multi_branch
        # initialize the encoders

        self._enc_types = OneHotEncoder(handle_unknown='ignore')
        self._enc_types.fit([[a] for a in range(n_types + 1)])

        self._enc_layers = OneHotEncoder(handle_unknown='ignore')
        self._enc_layers.fit([[a] for a in range(max_layers + 1)])

        self._enc_kernels = OneHotEncoder(handle_unknown='ignore')
        self._enc_kernels.fit([[a] for a in range(max_kernel + 1)])

    def encode(self, state):
        """Encode the state with one-hot encoding per position."""
        encoded_state = []
        for layer in state:
            e_type = self._enc_types.transform([[layer[1]]]).toarray()[0]
            e_kernel = self._enc_kernels.transform([[layer[2]]]).toarray()[0]
            e_pred1 = self._enc_layers.transform([[layer[3]]]).toarray()[0]
            if self.is_multi_branch:
                e_pred2 = self._enc_layers.transform([[layer[4]]]).toarray()[0]
            else:
                e_pred2 = []
            enc_layer = \
                list(e_type) + list(e_kernel) + list(e_pred1) + list(e_pred2)
            encoded_state.append(enc_layer)

        return np.array(encoded_state)
