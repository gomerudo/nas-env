"""Create a default environment for Neural Architecture Search."""

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from nas_gym.utl.miscellaneous import is_valid_config_file


class DefaultNASEnv(gym.Env):
    """Default Neural Architecture Search (NAS) environment."""

    metadata = {'render.modes': ['human']}

    def __init__(self, config_file=None, max_steps=100, dataset='meta-dataset',
                 is_learning=True):
        """Initialize the NAS environment, via a configuration file."""
        self.is_learning = is_learning
        self.max_steps = max_steps
        self.dataset = dataset

        # TODO: Think about the extra information we need
        self.info_dict = {}

        # TODO: Create a dataset handler that will take care of switching tasks
        self.dataset_handler = None
        self._config_file = config_file

        # TODO: set a default current_state (initial state) and call all
        # defaults setters.
        self.current_state = self.reset()
        self.step_count = 0
        # TODO: An attribute storing the current dataset to work on
        self.current_task = None

    def _set_all_from_default_config(self):
        if self._config_file is None:
            return self._load_defaults()

        return self._load_from_file()

    # TODO: stablish some very basic defaults.
    def _load_defaults(self):
        act_s = None
        obs_s = None
        # TODO: Load parameters from file

        # Finally, return the objects
        return act_s, obs_s

    def _load_from_file(self):
        act_s = None
        obs_s = None

        if not is_valid_config_file(self._config_file):
            raise ValueError(
                "Invalid configuration file. Please use a valid format."
            )
        # TODO: Load parameters from file

        # Finally, return the two objects
        return act_s, obs_s

    def step(self, action):
        """Perform an step in the environment, given an action."""
        # TODO:
        # 1. Step takes an action (a numeric value, I think...) and it will
        #   change a position in the observation_space's `Box`.
        self.current_state = ...  # The new architecture
        # 2. It will then build the neural network (TensorFlow) with the
        #   current state.
        nn = nas_helper.build_network(self.current_state)
        # 3. It will evaluate the neural network on the current_task (dataset)
        #   and will return the accuracy to be used as the reward.
        reward = nas_helper.train(nn)
        # 4. We return the tuple (state, reward, done, info)
        self.step_count += 1

        # Check whether or not we are done
        done_check = self.step_count - 1 == self.max_steps

        # Return the info
        return self.current_state, reward, done_check, self.info_dict

    def reset(self):
        """Reset the environment's state."""
        reset_state = None
        # TODO:
        # Reset will set the environment to the values specified in
        # `_load_default()` or `_load_from_file()`, accordingly.
        self._set_all_from_default_config()

        # TODO: Somehow build the reset_state (? I don't know how yet...)

        return reset_state

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


class NASHelper:

    @staticmethod
    def perform_action():
        """"""

    def train_network():
        """"""
