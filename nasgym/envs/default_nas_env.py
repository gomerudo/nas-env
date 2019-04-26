"""Create a default environment for Neural Architecture Search.

The default environment executes actions as in BlockQNN. In this module we
expose the main class `DefaultNASEnv`, together with two more helper classes:
`DefaultNASEnvParser` - in charge of parsing a YML with the definition of the
environment - and `NASEnvHelper` - with static methods that perform the main
computations and call methods from net_trainer mainly.

The NAS environment, makes use of OpenAI's Gym environment abstract class. To
extend more NAS environments, use this class as a model.
"""

import time
import numpy as np
import yaml
import gym
from gym import spaces
from nasgym.database.default_db import DefaultExperimentsDatabase
from nasgym.dataset_handlers.default_handler import AbstractDatasetHandler
from nasgym.net_ops.net_builder import LTYPE_ADD
from nasgym.net_ops.net_builder import LTYPE_AVGPOOLING
from nasgym.net_ops.net_builder import LTYPE_CONCAT
from nasgym.net_ops.net_builder import LTYPE_CONVULUTION
from nasgym.net_ops.net_builder import LTYPE_IDENTITY
from nasgym.net_ops.net_builder import LTYPE_MAXPOOLING
from nasgym.net_ops.net_builder import LTYPE_TERMINAL
from nasgym.net_ops.net_utils import sort_sequence
from nasgym.net_ops.net_trainer import EarlyStopNASTrainer
from nasgym.utl.miscellaneous import compute_str_hash
from nasgym.utl.miscellaneous import get_current_timestamp
from nasgym.utl.miscellaneous import infer_data_shape
from nasgym.utl.miscellaneous import infer_n_classes
from nasgym.utl.miscellaneous import is_valid_config_file
from nasgym.utl.miscellaneous import normalize_dataset
from nasgym.utl.miscellaneous import state_to_string


class DefaultNASEnv(gym.Env):
    """Default Neural Architecture Search (NAS) environment."""

    metadata = {'render.modes': ['human']}
    reward_range = (0.0, 100.0)

    def __init__(self, config_file="resources/nasenv.yml", max_steps=100,
                 max_layers=10, dataset_handler=None,
                 db_file="workspace/db_experiments.csv", log_path="workspace"):
        """Initialize the NAS environment, via a configuration file."""
        # 1. Assign the class' properties
        self.max_steps = max_steps
        self.max_layers = max_layers
        self.log_path = log_path

        # 2. Instanciate the database of experiments and its columns. Note that
        #    we never overwrite the file but append.
        self.db_manager = DefaultExperimentsDatabase(
            file_name=db_file,
            headers=[
                "dataset-nethash",
                "netstring",
                "reward",
                "timestamp",
                "running_time"
            ],
            pk_header="dataset-nethash",
            overwrite=False
        )

        # 3. Get the Gym spaces: observtions and actions
        self.observation_space, self.action_space, self.actions_info = \
            self._load_from_file(config_file)

        # 4. Validate and assign the dataset handler that will provide the
        #    image classification task to solve (this task might change
        #    between experiments)
        if dataset_handler is None:
            raise ValueError(
                "It is not possible to use a `None` dataset_handler. \
Please use a valid one."
            )

        if not isinstance(dataset_handler, AbstractDatasetHandler):
            raise TypeError(
                "Invalid type for dataset_handler. Use a class of type \
AbstractDatasetHandler"
            )
        self.dataset_handler = dataset_handler

        # 5. Reset the environment and the step count
        self.current_state = self.reset()
        self.step_count = 0

    def _load_from_file(self, config_file):
        # 1. Check the validity of the configuration file
        if not is_valid_config_file(config_file):
            raise ValueError(
                "Invalid configuration file. Please use a valid format."
            )

        # 2. Load parameters from file
        file_parser = DefaultNASEnvParser(config_file)

        # 3. Assign the desired return variables: the actual spaces
        act_s = file_parser.action_space
        act_info = file_parser.action_info
        obs_s = file_parser.observation_space

        # Finally, return the the spaces and the dictionary of actions.
        return obs_s, act_s, act_info

    def step(self, action):
        """Perform an step in the environment, given an action."""
        # 1. Perform the action: this will alter the internal state
        self.current_state = NASEnvHelper.perform_action(
            self.state,
            action,
            self.actions_info
        )
        # 2. We always sort the sequence
        self.current_state = sort_sequence(self.current_state)

        # 3. We build the composed ID: dataset_name + hash_of_sequence
        composed_id = "{d}-{h}".format(
            d=self.dataset_handler.current_dataset_name(),
            h=compute_str_hash(state_to_string(self.current_state))
        )

        # 4. Compute the reward: if it exists already in the DB, skip training,
        #    otherwise proceed to training.
        if self.db_manager.exists(composed_id):
            prev = self.db_manager.get_row(composed_id)
            reward = float(prev['reward'])
            running_time = int(prev['running_time'])

            print(
                "Skipping computationf of reward for {id} cause it has been \
found in the DB".format(id=composed_id)
            )
        else:
            start = time.time()
            reward = NASEnvHelper.reward(
                self.current_state,
                self.dataset_handler,
                self.log_path
            )
            end = time.time()
            running_time = int(end - start)
            # Fix the reward if they go outside the boundaries: Not really
            # needed but just to make sure...
            reward = DefaultNASEnv.reward_range[1] \
                if reward > DefaultNASEnv.reward_range[1] else reward
            reward = DefaultNASEnv.reward_range[0] \
                if reward < DefaultNASEnv.reward_range[0] else reward

            self.db_manager.add(
                {
                    "dataset-nethash": composed_id,
                    "netstring": state_to_string(self.current_state),
                    "reward": reward,
                    "timestamp": get_current_timestamp(),
                    "running_time": running_time,
                }
            )
            self.db_manager.save()

        # 5. Increase the number of steps cause we are done with the action
        self.step_count += 1

        # 6. Verify whether or not we are done: if we reached the max number of
        #    steps or if the action we performmed was a terminal action.
        done = self.step_count == self.max_steps or \
            NASEnvHelper.is_terminal(action, self.actions_info)

        # 7. Build additional information we want to return (as in gym.Env)
        info_dict = {
            'current_step': self.step_count,
            "dataset-nethash": composed_id,
            "running_time": running_time
        }

        # 8. Return the results as specified in gym.Env
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
        for row in self.current_state:
            print(row)

    # This is not from gym.Env interface. This is used by our Meta-RL algorithm
    def next_task(self):
        """Change the NN task by switching the dataset to be used."""
        # We always rely on the dataset handler, since the switching is
        # independent from the environment
        self.dataset_handler.next_dataset()


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
                    range(0, self.max_nlayers)

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


class NASEnvHelper:
    """Set of static methods that help the Default NAS environment."""

    @staticmethod
    def perform_action(space, action, action_info):
        """Perform an action in the environment's space."""
        # Always assert the action first
        NASEnvHelper.assert_valid_action(action, action_info)

        # Obtain the encoding
        encoding = NASEnvHelper.infer_action_encoding(action, action_info)

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

            # If found, assign the encoding to the available layer.
            if available_layer in range(0, space.shape[0]):
                space[available_layer] = \
                    np.array([available_layer + 1] + encoding)

        # Return the space
        return space

    @staticmethod
    def is_remove_action(action, action_info):
        """Check if an action is a 'remove' action."""
        NASEnvHelper.assert_valid_action(action, action_info)

        # Check if it is the terminal state
        return action_info[action].startswith("remove")

    @staticmethod
    def infer_action_encoding(action, action_info):
        """Obtain the encoding of an action, given its identifier in a dict."""
        NASEnvHelper.assert_valid_action(action, action_info)

        action_str = action_info[action]
        action_arr = action_str.split("_")
        print(action_arr)
        # Depending of the type of action...
        if NASEnvHelper.is_remove_action(action, action_info):
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
    def reward(state, dataset_handler, log_path="./workspace"):
        """Perform the training of the network, given (state, dataset) pair."""
        try:
            print("Obtaining train/validation sets")
            train_features, train_labels = dataset_handler.current_train_set()
            val_features, val_labels = dataset_handler.current_validation_set()

            hash_state = compute_str_hash(state_to_string(state))

            print("Obtaining the trainer")
            nas_trainer = EarlyStopNASTrainer(
                encoded_network=state,
                input_shape=infer_data_shape(train_features),
                n_classes=infer_n_classes(train_labels),
                batch_size=256,
                log_path="{lp}/trainer-{h}".format(lp=log_path, h=hash_state),
                mu=0.5,
                rho=0.5,
                variable_scope="cnn-{h}".format(h=hash_state)
            )
            print("Normalizing the training set")
            train_features = normalize_dataset(
                dataset=train_features,
                baseline=255
            )
            train_labels = train_labels.astype(np.int32)

            print("Training the network")
            nas_trainer.train(
                train_data=train_features,
                train_labels=train_labels,
                train_input_fn="default",
                n_epochs=12  # As specified by BlockQNN
            )

            print("Normalizing the validaation set")
            val_features = normalize_dataset(
                dataset=val_features,
                baseline=255
            )
            val_labels = val_labels.astype(np.int32)

            print("Evaluating the validation set")
            res = nas_trainer.evaluate(
                eval_data=val_features,
                eval_labels=val_labels,
                eval_input_fn="default"
            )

            print("Reporting the accuracy")
            accuracy = res['accuracy']
            # Compute the refined reward as defined
            reward = accuracy*100 - nas_trainer.weighted_log_density - \
                nas_trainer.weighted_log_flops

            return reward
        except Exception as ex:  # pylint: disable=broad-except
            # TODO: Make sure exceptions are printed correctly.
            print("Reward computation failed with exception:", ex)
            return 0.

    @staticmethod
    def is_terminal(action, action_info):
        """Check if an action induces a terminal state."""
        # Assert first
        NASEnvHelper.assert_valid_action(action, action_info)

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
