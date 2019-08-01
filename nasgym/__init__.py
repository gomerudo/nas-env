"""Register the different NAS environments we makea available by default."""

import logging
# import tensorflow as tf
from gym.envs.registration import register
import nasgym.utl.configreader as cr
from nasgym.dataset_handlers.default_handler import DefaultDatasetHandler

# Import the configuration fron the .ini file
CONFIG_INI = cr.read_configfile()
print("Parameters in configuration file are:", CONFIG_INI)

# Assign logger the specified name
nas_logger = logging.getLogger(CONFIG_INI[cr.SEC_DEFAULT][cr.PROP_LOGGER_NAME])

# Assign debug level
NAS_LOGGER_LEVEL = CONFIG_INI[cr.SEC_DEFAULT][cr.PROP_LOGGER_LEVEL]
nas_logger.setLevel(NAS_LOGGER_LEVEL)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(NAS_LOGGER_LEVEL)

# Set format
ch.setFormatter(
    logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
)
# Assign the stream handler to the logger
nas_logger.addHandler(ch)

# (train_data, train_labels), (eval_data, eval_labels) = \
#     tf.keras.datasets.cifar10.load_data()

# handler = DefaultDatasetHandler(
#     train_data, train_labels, eval_data, eval_labels, "cifar10"
# )

# config_file = "nas-env/resources/nasenv.yml"

register(
    id='NAS_cifar10-v1',
    entry_point='nasgym.envs:DefaultNASEnv',
    # kwargs={
    #     'config_file': config_file,
    #     'dataset_handler': handler
    # }
)
