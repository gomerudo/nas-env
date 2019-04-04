"""Create a default environment for Neural Architecture Search."""

import numpy as np
import yaml
import gym
from gym import spaces
from nas_gym.utl.miscellaneous import is_valid_config_file
from nas_gym.net_ops.net_builder import LTYPE_ADD
from nas_gym.net_ops.net_builder import LTYPE_AVGPOOLING
from nas_gym.net_ops.net_builder import LTYPE_CONCAT
from nas_gym.net_ops.net_builder import LTYPE_CONVULUTION
from nas_gym.net_ops.net_builder import LTYPE_IDENTITY
from nas_gym.net_ops.net_builder import LTYPE_MAXPOOLING
from nas_gym.net_ops.net_builder import LTYPE_TERMINAL


class DefaultNASEnv(gym.Env):
    """Default Neural Architecture Search (NAS) environment."""

    metadata = {'render.modes': ['human']}
    reward_range = (0.0, float('inf'))

    def __init__(self, config_file="./nasenv.yml", max_steps=100,
                 max_layers=10, dataset='meta-dataset', is_learning=True):
        """Initialize the NAS environment, via a configuration file."""
        self.is_learning = is_learning
        self.max_steps = max_steps
        self.dataset = dataset
        self.max_layers = max_layers

        # Get the spaces
        self.observation_space, self.action_space, self.actions_info = \
            self._load_from_file(config_file)

        # TODO: Create a dataset handler that will take care of switching tasks
        self.dataset_handler = None

        # TODO: set a default current_state (initial state) and call all
        # defaults setters.
        self.current_state = self.reset()
        self.step_count = 0

        # TODO: An attribute storing the current dataset to work on
        self.current_task = None

    def _load_from_file(self, config_file):
        act_s = None
        obs_s = None

        if not is_valid_config_file(config_file):
            raise ValueError(
                "Invalid configuration file. Please use a valid format."
            )

        # Load parameters from file
        file_parser = DefaultNASEnvParser(config_file)

        # Assign the desired return variables
        act_s = file_parser.action_space
        act_info = file_parser.action_info
        obs_s = file_parser.observation_space

        # Finally, return the two objects
        return obs_s, act_s, act_info

    def step(self, action):
        """Perform an step in the environment, given an action."""
        # TODO:
        # 1. Step takes an action (a numeric value, I think...) and it will
        #   change a position in the observation_space's `Box`.
        self.current_state = NASHelper.perform_action(
            self.state,
            action,
            self.actions_info
        )
        # 2. It will then build the neural network (TensorFlow) with the
        #   current state.
        reward = NASHelper.train_network(self.current_state, self.current_task)
        # 3. It will evaluate the neural network on the current_task (dataset)
        #   and will return the accuracy to be used as the reward.

        # 4. We return the tuple (state, reward, done, info)
        self.step_count += 1
        done = self.step_count == self.max_steps or \
            NASHelper.is_terminal(action, self.actions_info)

        # Check whether or not we are done
        # done_check = self.step_count - 1 == self.max_steps

        # TODO: uild the the info
        info_dict = {}

        return self.current_state, reward, done, info_dict

    def reset(self):
        """Reset the environment's state."""
        # Reset the state to only zeros
        self.state = np.zeros(
            shape=self.observation_space.shape,
            dtype=np.int32
        )

        return self.state

    def render(self, mode='human'):
        """Render the environment, according to the specified mode."""
        # TODO: think about how to render the network, maybe a tensorflow plot
        #   or simply the vector as text.
        print(self.current_state)

    # This is not from gym.Env interface. This is used by our Meta-RL algorithm
    def new_task(self):
        """Change the NN task by switching the dataset to be used."""
        if self.dataset == 'meta-dataset':  # TODO: Change as global variable
            self.current_task = self.dataset_handler.next_dataset()


class DefaultNASEnvParser:
    """Default parser."""

    def __init__(self, config_file):
        """Constructor."""
        self.config_file = config_file

        # Perform only validations
        if not is_valid_config_file(self.config_file):
            raise ValueError(
                "Invalid configuration file. Please use a valid format."
            )
        # Load file
        self.reload_file()

    def reload_file(self):
        """Load the config file from scratch."""
        with open(self.config_file, 'r') as yaml_file:
            content = yaml.load(yaml_file)
            # Validate the content
            self._nasenv_dict = self._validate_and_read(content)

        # Assign the important properties
        self.config_name = self._nasenv_dict['name']
        self.config_version = self._nasenv_dict['version']
        self.max_nlayers = self._nasenv_dict['max_nlayers']

        # Load the action space and the configuration space
        self.action_space, self.action_info = self._populate_action_space()
        self.observation_space = self._populate_observation_space()

    def _populate_action_space(self):
        action_info = {}
        counter = 0

        for layer in self._nasenv_dict['layers']:
            for layer_key, layer_config in layer.items():
                self._validate_layer_config(layer_config)

                # Check Kernel Size
                kernels_list = [0] if layer_config['kernel_size'] is None else\
                    layer_config['kernel_size']

                pred1_list = [0] if not layer_config['pred1'] else \
                    range(0, self.max_nlayers + 1)

                pred2_list = [0] if not layer_config['pred2'] else \
                    range(0, self.max_nlayers + 1)

                # For every kernel
                for kernel in kernels_list:
                    # For every predecesor1
                    for c_pred1 in pred1_list:
                        # For every predecesor2
                        for c_pred2 in pred2_list:
                            action_type = \
                                "{type}_k-{kernel}_pred1-{pred1}_pred2-\
{pred2}".format(type=layer_key, kernel=kernel, pred1=c_pred1, pred2=c_pred2)
                            action_info[counter] = action_type
                            counter += 1

        # At the end, add REMOVE actions
        for layer in range(self.max_nlayers):
            action_info[counter] = "remove_{l}".format(l=layer)
            counter += 1

        # The actual set of actions
        # action_set = list(range(counter))
        return spaces.Discrete(counter), action_info

    def _populate_observation_space(self):
        return spaces.Box(
            0,
            np.inf,
            shape=[self.max_nlayers, 5],  # Default length per NSC is 5
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


class NASHelper:
    """Set of static methods that help the Default NAS environment."""

    @staticmethod
    def perform_action(space, action, action_info):
        """Perform an action in the environment's space."""
        # Always assert the action first
        NASHelper.assert_valid_action(action, action_info)

        # Obtain the encoding
        encoding = NASHelper.infer_action_encoding(action, action_info)

        # Perform the modification in the space, given the type of action

        if isinstance(encoding, int):  # It is a remove action
            space[encoding] = np.zeros(5)
        elif isinstance(encoding, list):  # It is just another action
            # Search an available space
            available_layer = -1
            for i, layer in enumerate(space):
                if not layer[0]:
                    available_layer = i
                    break

            # If found, return the 
            if available_layer in range(0, space.shape[0]):
                space[available_layer] = \
                    np.array([available_layer + 1] + encoding)

        return space

    @staticmethod
    def is_remove_action(action, action_info):
        """Check if an action is a 'remove' action."""
        NASHelper.assert_valid_action(action, action_info)

        # Check if it is the terminal state
        return action_info[action].startswith("remove")

    @staticmethod
    def infer_action_encoding(action, action_info):
        """Obtain the encoding of an action, given its identifier in a dict."""
        NASHelper.assert_valid_action(action, action_info)

        action_str = action_info[action]
        action_arr = action_str.split("_")
        print(action_arr)
        # Depending of the type of action...
        if NASHelper.is_remove_action(action, action_info):
            # Return only the number of the layer to remove
            return action_arr[1]

        # Otherwise ...
        layer_type = action_arr[0]
        layer_kernel_size = action_arr[1].split("-")[1]
        layer_pred1 = action_arr[2].split("-")[1]
        layer_pred2 = action_arr[3].split("-")[1]

        # Option 2: use a dictionary and just do layer_type=dict[type]
        if layer_type == "convolution":
            layer_type = LTYPE_CONVULUTION
        if layer_type == "maxpooling":
            layer_type = LTYPE_MAXPOOLING
        if layer_type == "avgpooling":
            layer_type = LTYPE_AVGPOOLING
        if layer_type == "identity":
            layer_type = LTYPE_IDENTITY
        if layer_type == "add":
            layer_type = LTYPE_ADD
        if layer_type == "concat":
            layer_type = LTYPE_CONCAT
        if layer_type == "terminal":
            layer_type = LTYPE_TERMINAL

        return [layer_type, layer_kernel_size, layer_pred1, layer_pred2]

    @staticmethod
    def train_network(state, dataset):
        """Perform the training of the network, given (state, dataset) pair."""
        return 0.  # For now... TODO: implement it!

    @staticmethod
    def is_terminal(action, action_info):
        """Check if an action induces a terminal state."""
        # Assert first
        NASHelper.assert_valid_action(action, action_info)

        # Check if it is the terminal state
        return action_info[action].startswith("terminal")

    @staticmethod
    def assert_valid_action(action, action_info):
        """Whether or not the action is a valid one."""
        try:
            _ = action_info[action]
        except:
            raise RuntimeError(
                "Invalid action. Valid actions are: {dict}".format(
                    dict=action_info
                )
            )
