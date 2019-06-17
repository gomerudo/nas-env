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
import gym
import nasgym.utl.configreader as cr
from nasgym import nas_logger
from nasgym import CONFIG_INI
from nasgym.envs.envspecs_parsers import AbstractEnvSpecsParser
from nasgym.envs.factories import EnvSpecsParserFactory
from nasgym.envs.factories import DatasetHandlerFactory
from nasgym.envs.factories import TrainerFactory
from nasgym.database.default_db import DefaultExperimentsDatabase
from nasgym.dataset_handlers.default_handler import AbstractDatasetHandler
from nasgym.dataset_handlers.default_handler import DefaultDatasetHandler
from nasgym.dataset_handlers.metadataset_handler import MetaDatasetHandler
from nasgym.net_ops.net_builder import LTYPE_ADD
from nasgym.net_ops.net_builder import LTYPE_AVGPOOLING
from nasgym.net_ops.net_builder import LTYPE_CONCAT
from nasgym.net_ops.net_builder import LTYPE_CONVULUTION
from nasgym.net_ops.net_builder import LTYPE_IDENTITY
from nasgym.net_ops.net_builder import LTYPE_MAXPOOLING
from nasgym.net_ops.net_builder import LTYPE_TERMINAL
from nasgym.net_ops.net_utils import sort_sequence
from nasgym.utl.miscellaneous import compute_str_hash
from nasgym.utl.miscellaneous import get_current_timestamp
from nasgym.utl.miscellaneous import state_to_string
from nasgym.utl.miscellaneous import get_current_layer


class DefaultNASEnv(gym.Env):
    """Default Neural Architecture Search (NAS) environment."""

    metadata = {'render.modes': ['human']}
    reward_range = (0.0, 100.0)

    def __init__(self, config_file="resources/nasenv.yml", max_steps=100,
                 dataset_handler="default", action_space_type="default",
                 db_file="workspace/db_experiments.csv", log_path="workspace",
                 **kwargs):
        """Initialize the NAS environment, via a configuration file."""
        nas_logger.debug("Creating instance of the Default NAS environment")

        # 1. Try to set each of the properties from config.ini or arguments
        self._read_params_from_config(
            max_steps=max_steps,
            log_path=log_path,
            db_file=db_file,
            config_file=config_file,
            action_space_type=action_space_type,
            dataset_handler=dataset_handler
        )

        # 2. Instanciate the database of experiments and its columns. Note that
        #    we never overwrite the file but append.
        self.db_manager = DefaultExperimentsDatabase(
            file_name=self.db_file,
            headers=[
                "dataset-nethash",
                "netstring",
                "reward",
                "accuracy",
                "density",
                "flops",
                "timestamp",
                "running_time",
                "is_valid",
            ],
            pk_header="dataset-nethash",
            overwrite=False
        )

        # 3. Get the Gym spaces: observtions and actions
        if isinstance(self.action_space_type, AbstractEnvSpecsParser):
            space_parser = self.action_space_type
        elif isinstance(self.action_space_type, str):
            space_parser = EnvSpecsParserFactory.get_parser(
                config_file=self.config_file,
                parser_type=self.action_space_type
            )
        else:
            raise TypeError("Invalid action_space_type")

        # Load the configuration and assign the values, then export them
        space_parser.reload_configuration()
        self.observation_space = space_parser.observation_space
        self.action_space, self.actions_info = space_parser.action_space
        self._export_actions_info()

        # 4. Validate and assign the dataset handler that will provide the
        #    image classification task to solve (this task might change
        #    between experiments)
        if isinstance(self.dataset_handler, AbstractDatasetHandler):
            pass  # It is already assigned
        elif isinstance(self.dataset_handler, str):
            self.dataset_handler = DatasetHandlerFactory.get_handler(
                handler_type=self.dataset_handler, **kwargs
            )
        else:
            raise TypeError("Invalid type for dataset_handler")

        # 5. Reset the environment: a) set initial status as matrix of 0s, b)
        #    set the step_count to 0, set the predecessor*_shift to 0.
        self.state = self.reset()

    def _read_params_from_config(self, max_steps, log_path, db_file,
                                 config_file, action_space_type,
                                 dataset_handler):
        # The max_steps
        try:
            self.max_steps = \
                CONFIG_INI[cr.SEC_NASENV_DEFAULT][cr.PROP_MAXSTEPS]
        except KeyError:
            self.max_steps = max_steps

        # The log_path
        try:
            self.log_path = CONFIG_INI[cr.SEC_DEFAULT][cr.PROP_LOGPATH]
        except KeyError:
            self.log_path = log_path

        # The db_file
        try:
            self.db_file = CONFIG_INI[cr.SEC_NASENV_DEFAULT][cr.PROP_DBFILE]
        except KeyError:
            self.db_file = db_file

        # The config_file (.yml)
        try:
            self.config_file = \
                CONFIG_INI[cr.SEC_NASENV_DEFAULT][cr.PROP_CONFIGFILE]
        except KeyError:
            self.config_file = config_file

        # The action_space_type
        try:
            self.action_space_type = \
                CONFIG_INI[cr.SEC_NASENV_DEFAULT][cr.PROP_ACTION_SPACE_TYPE]
        except KeyError:
            self.action_space_type = action_space_type

        # The dataset handler
        try:
            self.dataset_handler = \
                CONFIG_INI[cr.SEC_NASENV_DEFAULT][cr.PROP_DATASET_HANDLER]
        except KeyError:
            self.dataset_handler = dataset_handler

    def _export_actions_info(self):
        actions_info_db = DefaultExperimentsDatabase(
            file_name="{root}/{name}".format(
                root=self.log_path,
                name="actions_info.csv"
            ),
            headers=[
                "id",
                "action",
            ],
            pk_header="id",
            overwrite=True
        )

        for key, value in self.actions_info.items():
            actions_info_db.add({'id': key, 'action': value})
        actions_info_db.save()

    def save_db_experiments(self):
        self.db_manager.save()

    def step(self, action):
        """Perform an step in the environment, given an action."""
        pred_oob = False  # Default: False. Value can change in the first 'if'
        bkp_original = self.state.copy()  # The state before any action

        # 1. Perform the action so we can act accordingly.
        action_res = NASEnvHelper.perform_action(
            self.state,
            action,
            self.actions_info,
            self.pred1_shift,
            self.pred2_shift
        )

        # Option a) Shift the predecessors
        if isinstance(action_res, tuple):
            if action_res[1] == 'U':
                shift = 1
            if action_res[1] == 'D':
                shift = -1
            if action_res[0] == 1:
                self.pred1_shift += shift
            if action_res[0] == 2:
                self.pred2_shift += shift

            # Check if predecessors are out of boundaries after the shift
            pred_oob = not NASEnvHelper.is_valid_pred_shift(
                self.pred1_shift, self.pred2_shift, self.state
            )
        # Option b) The action was 'add layer', so we just change the state
        else:
            self.state = action_res

        # 2. We always sort the state representation
        self.state = sort_sequence(self.state, as_list=False)

        # 3. We build the composed ID: dataset_name + hash_of_sequence
        composed_id = "{d}-{h}".format(
            d=self.dataset_handler.current_dataset_name(),
            h=compute_str_hash(state_to_string(self.state))
        )

        # 4. Compute the reward: if it exists already in the DB, skip
        #    training, otherwise proceed to training.
        if self.db_manager.exists(composed_id):
            nas_logger.info(
                "Skipping reward computation for architecture %s cause it \
already exists the DB of experiments", composed_id
            )

            prev = self.db_manager.get_row(composed_id)
            reward = float(prev['reward'])
            accuracy = float(prev['accuracy'])
            density = float(prev['density'])
            flops = float(prev['flops'])
            running_time = int(prev['running_time'])
            status = int(prev['is_valid'])
        else:
            nas_logger.info(
                "Reward for architecture %s has not been found in the DB",
                composed_id
            )

            start = time.time()
            reward, accuracy, density, flops, status = NASEnvHelper.reward(
                self.state,
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
                    "netstring": self.state,
                    "reward": reward,
                    "accuracy": accuracy,
                    "density": density,
                    "flops": flops,
                    "timestamp": get_current_timestamp(),
                    "running_time": running_time,
                    "is_valid": status
                }
            )

        # If the predecessors were out of boundaries, the reward is 0.
        reward = 0 if pred_oob else reward

        # A. Increase the number of steps cause we are done with the action
        self.step_count += 1

        # B. Verify whether or not we are done: if we reached the max number of
        #    steps or if the action we performmed was a terminal action.
        #    Additionally, we consider 'dead' an invalid layer (cause the next
        #    ones will always be invalid too.)
        done = self.step_count == self.max_steps or \
            NASEnvHelper.is_terminal(action, self.actions_info) or not status \
            or self.state[0, 0] != 0 or pred_oob  # first 'or' means full

        if NASEnvHelper.is_predecessor_action(action, self.actions_info):
            inferred = NASEnvHelper.infer_action_predecessor_encoding(
                action, self.actions_info
            )
        else:
            inferred = NASEnvHelper.infer_action_encoding(
                action,
                self.actions_info,
                get_current_layer(bkp_original),
                self.pred1_shift,
                self.pred2_shift,
            )
        # C. Build additional information we want to return (as in gym.Env)
        info_dict = {
            "step_count": self.step_count,
            "valid": status,
            "composed_id": composed_id,
            "original_state": bkp_original,
            "original_state_hashed": compute_str_hash(
                state_to_string(bkp_original)
            ),
            "end_state": self.state,
            "end_state_hashed": compute_str_hash(
                state_to_string(self.state)
            ),
            "action_id": action,
            "action_inferred": inferred,
            "reward": reward,
            "done": done,
            "running_time": running_time,
            "pred1_shift": self.pred1_shift,
            "pred2_shift": self.pred2_shift, 
        }

        # D. Return the results as specified in gym.Env
        return self.state, reward, done, info_dict

    def reset(self):
        """Reset the environment's state."""
        # Reset the state to only zeros
        nas_logger.debug("Resetting environment")

        self.state = np.zeros(
            shape=self.observation_space.shape,
            dtype=np.int32
        )
        self.step_count = 0

        # Two variables to shift the position of the predecessor
        self.pred1_shift = 0
        self.pred2_shift = 0

        return self.state

    def render(self, mode='human'):
        """Render the environment, according to the specified mode."""
        for row in self.state:
            print(row)

    # This is not from gym.Env interface. This is used by our Meta-RL algorithm
    def next_task(self):
        """Change the NN task by switching the dataset to be used."""
        nas_logger.debug("Switching to next task in the dataset handler")
        # We always rely on the dataset handler, since the switching is
        # independent from the environment
        self.dataset_handler.next_dataset()


class NASEnvHelper:
    """Set of static methods that help the Default NAS environment."""

    @staticmethod
    def perform_action(space, action, action_info, shift1=0, shift2=0):
        """Perform an action in the environment's space."""
        # Always assert the action first
        target = space.copy()
        NASEnvHelper.assert_valid_action(action, action_info)

        # Handle the case where the predecessor are altered
        if NASEnvHelper.is_predecessor_action(action, action_info):
            return NASEnvHelper.infer_action_predecessor_encoding(
                action, action_info
            )

        # If it is not a predecessor action, obtain the encoding...
        encoding = NASEnvHelper.infer_action_encoding(
            action, action_info, get_current_layer(space), shift1, shift2
        )

        nas_logger.debug(
            "Performing action %d, inferred as %s", action, encoding
        )
        # Perform the modification in the space, given the type of action
        if isinstance(encoding, int):  # It is a remove action
            target[encoding] = np.zeros(5)
        elif isinstance(encoding, list):  # It is just another action
            # Search an available row
            available_row = -1
            for i, layer in enumerate(target):
                if not layer[0]:
                    available_row = i
                    break

            # If found, assign the encoding to the available layer.
            if available_row in range(0, target.shape[0]):
                target[available_row] = \
                    np.array([np.amax(target[:, 0], axis=0) + 1] + encoding)

        # Return the target space
        return target

    @staticmethod
    def is_remove_action(action, action_info):
        """Check if an action is a 'remove' action."""
        NASEnvHelper.assert_valid_action(action, action_info)

        # Check if it is the terminal state
        return action_info[action].startswith("remove")

    @staticmethod
    def infer_action_encoding(action, action_info, current_layer=0, shift1=0,
                              shift2=0):
        """Obtain the encoding of an action, given its identifier in a dict."""
        NASEnvHelper.assert_valid_action(action, action_info)

        action_str = action_info[action]
        action_arr = action_str.split("_")

        # Otherwise ...
        layer_type = action_arr[0]
        layer_kernel_size = int(action_arr[1].split("-")[1])
        layer_pred1 = action_arr[2].split("-")[1]
        layer_pred2 = action_arr[3].split("-")[1]

        if layer_pred1 == "L":
            layer_pred1 = current_layer - shift1
        else:
            layer_pred1 = int(layer_pred1)

        if layer_pred2 == "BL":
            layer_pred2 = 0 if not current_layer else current_layer - shift2
            # layer_pred2 = 0 if not current_layer else current_layer - 1
        else:
            layer_pred2 = int(layer_pred2)

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
    def infer_action_predecessor_encoding(action, action_info):
        """Obtain the encoding of an action, given its identifier in a dict."""
        NASEnvHelper.assert_valid_action(action, action_info)

        # predecessor_p-X_op-O
        action_str = action_info[action]
        action_arr = action_str.split("_")  # [predecessor, p-X, op-O]

        # Otherwise ...
        predecessor = int(action_arr[1].split("-")[1])
        operation = action_arr[2].split("-")[1]

        return (predecessor, operation)  # Return a tuple

    @staticmethod
    def reward(state, dataset_handler, log_path="./workspace"):
        """Perform the training of the network, given (state, dataset) pair."""
        try:

            hash_state = compute_str_hash(state_to_string(state))
            composed_id = "{d}-{h}".format(
                d=dataset_handler.current_dataset_name(), h=hash_state
            )

            nas_logger.debug(
                "Reward of architecture %s will be computed", composed_id
            )

            try:
                trainer_type = \
                    CONFIG_INI[cr.SEC_NASENV_DEFAULT][cr.PROP_TRAINER_TYPE]
            except KeyError:
                trainer_type = "early-stop"

            nas_trainer = TrainerFactory.get_trainer(
                trainer_type=trainer_type,
                state=state.copy(),
                dataset_handler=dataset_handler,
                log_path="{lp}/trainer-{h}".format(lp=log_path, h=composed_id),
                variable_scope="cnn-{h}".format(h=hash_state)
            )

            nas_logger.debug(
                "Trainer used for reward computation is %s. Object's \
attributes are:", type(nas_trainer)
            )

            nas_logger.debug("  input_shape: %s", nas_trainer.input_shape)
            nas_logger.debug("  batch_size: %s", nas_trainer.batch_size)
            nas_logger.debug("  log_path: %s", nas_trainer.log_path)
            nas_logger.debug("  mu: %s", nas_trainer.mu)
            nas_logger.debug("  rho: %s", nas_trainer.rho)
            nas_logger.debug(
                "  variable_scope: %s", nas_trainer.variable_scope
            )

            # Obtain values to passs to train/evaluate functions.
            try:
                n_epochs = CONFIG_INI[cr.SEC_TRAINER_DEFAULT][cr.PROP_NEPOCHS]
                nas_logger.debug("Using n_epochs from config.ini")
            except KeyError:
                n_epochs = 12

            if isinstance(dataset_handler, MetaDatasetHandler):
                train_features, train_labels = None, None
                val_features, val_labels = None, None

                def custom_train_input_fn():
                    return dataset_handler.current_train_set()

                def custom_eval_input_fn():
                    return dataset_handler.current_validation_set()

                train_input_fn = custom_train_input_fn
                eval_input_fn = custom_eval_input_fn

            if isinstance(dataset_handler, DefaultDatasetHandler):
                train_features, train_labels = \
                    dataset_handler.current_train_set()
                val_features, val_labels = \
                    dataset_handler.current_validation_set()
                train_input_fn = "default"
                eval_input_fn = "default"

            nas_logger.debug(
                "Training architecture %s for %d epochs", composed_id, n_epochs
            )
            nas_trainer.train(
                train_data=train_features,
                train_labels=train_labels,
                train_input_fn=train_input_fn,
                n_epochs=n_epochs  # As specified by BlockQNN
            )

            nas_logger.debug("Evaluating architecture %s", composed_id)
            res = nas_trainer.evaluate(
                eval_data=val_features,
                eval_labels=val_labels,
                eval_input_fn=eval_input_fn
            )
            nas_logger.debug(
                "Train-evaluation procedure finished for architecture %s",
                composed_id
            )

            accuracy = res['accuracy']
            # Compute the refined reward as defined
            reward = accuracy*100 - nas_trainer.weighted_log_density - \
                nas_trainer.weighted_log_flops

            return reward, accuracy, nas_trainer.density, nas_trainer.flops, \
                True
        except Exception as ex:  # pylint: disable=broad-except
            nas_logger.info(
                "Reward computation of architecture %s failed with exception \
of type %s. Message is: %s", composed_id, type(ex), str(ex)
            )
            return 0., 0., 0., 0., False

    @staticmethod
    def is_terminal(action, action_info):
        """Check if an action induces a terminal state."""
        # Assert first
        NASEnvHelper.assert_valid_action(action, action_info)

        # Check if it is the terminal state
        return action_info[action].startswith("terminal")

    @staticmethod
    def is_predecessor_action(action, action_info):
        """Check if an action induces a terminal state."""
        # Assert first
        NASEnvHelper.assert_valid_action(action, action_info)

        # Check if it is the terminal state
        return action_info[action].startswith("predshift")

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

    @staticmethod
    def is_valid_pred_shift(shift1, shift2, state):
        """Test if a shift is valid, given the current state."""
        n_layers = state.shape[0]
        current_layer = state[n_layers - 1][0]

        shifted1 = current_layer + shift1
        shifted2 = current_layer + shift2

        return (shifted1 >= 0 and shifted1 <= current_layer) and \
               (shifted2 >= 0 and shifted2 <= current_layer)
