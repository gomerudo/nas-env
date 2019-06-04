"""Define helper functions for initialization of default_nas_env classes."""

from abc import ABC, abstractmethod
import yaml
from gym import spaces
import numpy as np
from nasgym.utl.miscellaneous import is_valid_config_file
from nasgym import nas_logger as logger


class AbstractEnvSpecsParser(ABC):
    """Abstract class for dataset handler."""

    def __init__(self, config_file):
        """General purpose constructor."""
        self.config_file = config_file
        super().__init__()

    @abstractmethod
    def reload_configuration(self):
        """Reload the configuration from the config_file."""

    @property
    @abstractmethod
    def observation_space(self):
        """Return the constructed."""

    @property
    @abstractmethod
    def action_space(self):
        """Return the action space and an info dictionary for the actions."""


class DefaultEnvSpecsParser(AbstractEnvSpecsParser):
    """Default parser for the environment specs."""

    def __init__(self, config_file):
        """Constructor."""
        super(DefaultEnvSpecsParser, self).__init__(config_file=config_file)
        self.reload_configuration()

    def reload_configuration(self):
        """Reload the configuration form the config_file."""
        if not is_valid_config_file(self.config_file):
            raise ValueError(
                "Invalid configuration file. Please use a valid format."
            )

        logger.debug(
            "Loading Default NAS configuration from file %s", self.config_file
        )
        # The re-loadded attributes
        self._observation_space, self._action_space, self._action_info = \
            self._load()

    def _load(self):
        logger.debug("Re-loading configuration file %s", self.config_file)
        with open(self.config_file, 'r') as yaml_file:
            content = yaml.load(yaml_file)
            # Validate the content
            nasenv_dict = self._validate_and_read(content)

        # Assign the important properties. We don't need the first two for now
        # config_name = nasenv_dict['name']
        # config_version = nasenv_dict['version']
        max_nlayers = nasenv_dict['max_nlayers']

        # Load the action space and the configuration space
        action_space, action_info = self._populate_action_space(
            max_nlayers, nasenv_dict
        )
        observation_space = self._populate_observation_space(max_nlayers)

        return observation_space, action_space, action_info

    def _populate_action_space(self, max_nlayers, nasenv_dict):
        logger.debug("Obtaining the action space for the environment")
        action_info = {}
        counter = 0

        for layer in nasenv_dict['layers']:
            for layer_key, layer_config in layer.items():
                self._validate_layer_config(layer_config)

                # Check Kernel Size
                kernels_list = [0] if layer_config['kernel_size'] is None else\
                    layer_config['kernel_size']

                pred1_list = [0] if not layer_config['pred1'] else \
                    range(0, max_nlayers)

                # For every kernel
                for kernel in kernels_list:
                    # For every predecesor1
                    for c_pred1 in pred1_list:
                        # For every predecesor2
                        pred2_list = [0] if not layer_config['pred2'] else \
                            range(0, c_pred1)
                        for c_pred2 in pred2_list:
                            action_type = \
                                "{type}_k-{kernel}_pred1-{pred1}_pred2-\
{pred2}".format(type=layer_key, kernel=kernel, pred1=c_pred1, pred2=c_pred2)
                            action_info[counter] = action_type
                            counter += 1

        return spaces.Discrete(counter), action_info

    def _populate_observation_space(self, max_nlayers):
        return spaces.Box(
            0,
            np.inf,
            shape=[max_nlayers, 5],  # Default length per NSC is 5
            dtype='int32'
        )

    def _validate_layer_config(self, layer):
        expected = set(['id', 'kernel_size', 'pred1', 'pred2'])
        observed_keys = set(layer.keys())

        if not expected <= observed_keys:
            raise RuntimeError(
                "Keys for layer {name} are invalid. Expected elements are: \
{ex}".format(name=layer, ex=expected)
            )

    def _validate_and_read(self, content):
        try:
            # Try to get the 'nasenv' object
            nasenv_dict = content['nasenv']

            # Validate the minimum set of expected keys is correct
            min_expected = set(['max_nlayers', 'layers', 'encoding'])
            observed_keys = set(nasenv_dict.keys())
            if not min_expected <= observed_keys:
                raise RuntimeError(
                    "Keys in YAML file are not the minimal expected for the \
NAS environment definition. Minimun expected is {me}".format(me=min_expected)
                )

            # Validate that we have valid max_n_layers, encoding and layers
            #   1. The max_nlayers
            try:
                nasenv_dict['max_nlayers'] = int(nasenv_dict['max_nlayers'])
            except TypeError:
                raise RuntimeError("Property max_n_layers is not an integer")
            #   2. The encoding is valid
            if nasenv_dict['encoding'] != "NSC":
                raise RuntimeError("Invalid encoding for configuration file.")

            if not isinstance(nasenv_dict['layers'], list):
                raise RuntimeError("Layers must be a valid .")
            # Finally, assign the properties to the object
            #   1. the configuration name
            try:
                _ = nasenv_dict['name']
            except KeyError:
                nasenv_dict['name'] = "Unknown NAS Configuration"
            #   2. the configuration version
            try:
                _ = str(nasenv_dict['version'])
            except KeyError:
                nasenv_dict['version'] = "1.0"

            return nasenv_dict
        except KeyError:
            raise RuntimeError(
                "Invalid config file. Does not contain a 'nasenv' definition"
            )

    @property
    def observation_space(self):
        """Return the observation space."""
        return self._observation_space

    @property
    def action_space(self):
        """Return the action space and its info dictionary."""
        return self._action_space, self._action_info
