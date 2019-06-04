"""Read the configuration file (.ini) with properties of the environment."""

import os
import configparser

SEC_DEFAULT = "DEFAULT"
SEC_NASENV_DEFAULT = "nasenv.default"
SEC_TRAINER_DEFAULT = "trainer.default"
SEC_TRAINER_EARLYSTOP = "trainer.earlystop"
SEC_TRAINER_TENSORFLOW = "trainer.tensorflow"
SEC_METADATASET = "metadataset"

# Default
PROP_LOGPATH = "LogPath"
PROP_LOGGER_LEVEL = "LoggerLevel"
PROP_LOGGER_NAME = "LoggerName"

# Default NASEnv
PROP_CONFIGFILE = "ConfigFile"
PROP_MAXSTEPS = "MaxSteps"
PROP_DBFILE = "DbFile"
PROP_ACTION_SPACE_TYPE = "DatasetHandler"
PROP_DATASET_HANDLER = "ActionSpaceType"

# Trainer Default
PROP_BATCHSIZE = "BatchSize"
PROP_NEPOCHS = "NEpochs"
PROP_OPTIMIZER_LEARNINGRATE = "OptimizerLearningRate"
PROP_OPTIMIZER_BETA1 = "OptimizerBeta1"
PROP_OPTIMIZER_BETA2 = "OptimizerBeta2"
PROP_OPTIMIZER_EPSILON = "OptimizerEpsilon"
PROP_FCLUNITS = "FCLUnits"
PROP_DROPOUTLAYER_RATE = "DropoutLayerRate"

# TensorFlow
PROP_ENABLE_DISTRIBUTED = "EnableDistributed"
PROP_ENABLE_DEVICEPLACEMENT = "EnableLogDevicePlacement"
PROP_ALLOW_MEMORYGROWTH = "AllowMemoryGrowth"

# Trainer Early Stop
PROP_MUWEIGHT = "MuWeight"
PROP_RHOWEIGHT = "RhoWeight"

# Metadataset
PROP_TFRECORDS_ROOTDIR = "TFRecordsRootDir"
PROP_DATASETID = "DatasetID"
PROP_TRAIN_TEST_SPLIT_PROP = "TrainTestSplitProp"
PROP_RANDOMSEED = "RandomSeed"


# TODO: handle defaults
def read_configfile():
    """Read the configuration file from a very specific location."""
    # Read the config file
    config = configparser.ConfigParser()
    expected_location = os.getenv(
        'NAS_DMRL_CONFIG_FILE', 'resources/config.ini'
    )
    print("Reading configuration file from", expected_location)

    config.read(expected_location)

    # Prepare the resulting dictionary
    res = {}
    res[SEC_DEFAULT] = {}
    res[SEC_NASENV_DEFAULT] = {}
    res[SEC_TRAINER_DEFAULT] = {}
    res[SEC_TRAINER_EARLYSTOP] = {}
    res[SEC_TRAINER_TENSORFLOW] = {}
    res[SEC_METADATASET] = {}

    # We are interested in five sections. Since we need the configuration to be
    # valid, we hardcode the parsing. When the properties/sections are not in
    # the file, we force default values to avoid errors in the experiments.

    # SECTION 1: Defaults
    # defaults = config.defaults()
    if PROP_LOGPATH in config[SEC_DEFAULT]:
        _set_property(SEC_DEFAULT, PROP_LOGPATH, config, res, None)

    if PROP_LOGGER_LEVEL in config[SEC_DEFAULT]:
        _set_property(SEC_DEFAULT, PROP_LOGGER_LEVEL, config, res, int)

    if PROP_LOGGER_NAME in config[SEC_DEFAULT]:
        _set_property(SEC_DEFAULT, PROP_LOGGER_NAME, config, res, None)

    # SECTION 2: Default NASEnv
    if _section_exists(SEC_NASENV_DEFAULT, config):
        if PROP_CONFIGFILE in config[SEC_NASENV_DEFAULT]:
            _set_property(
                SEC_NASENV_DEFAULT,
                PROP_CONFIGFILE,
                config,
                res,
                None
            )

        if PROP_MAXSTEPS in config[SEC_NASENV_DEFAULT]:
            _set_property(
                SEC_NASENV_DEFAULT,
                PROP_MAXSTEPS,
                config,
                res,
                int
            )

        if PROP_DBFILE in config[SEC_NASENV_DEFAULT]:
            _set_property(SEC_NASENV_DEFAULT, PROP_DBFILE, config, res, None)

        if PROP_ACTION_SPACE_TYPE in config[SEC_NASENV_DEFAULT]:
            _set_property(
                SEC_NASENV_DEFAULT,
                PROP_ACTION_SPACE_TYPE,
                config,
                res,
                None
            )

        if PROP_DATASET_HANDLER in config[SEC_NASENV_DEFAULT]:
            _set_property(
                SEC_NASENV_DEFAULT,
                PROP_DATASET_HANDLER,
                config,
                res,
                None
            )

    # SECTION 3: Default trainer
    if _section_exists(SEC_TRAINER_DEFAULT, config):
        if PROP_BATCHSIZE in config[SEC_TRAINER_DEFAULT]:
            _set_property(
                SEC_TRAINER_DEFAULT,
                PROP_BATCHSIZE,
                config,
                res,
                int
            )

        if PROP_NEPOCHS in config[SEC_TRAINER_DEFAULT]:
            _set_property(SEC_TRAINER_DEFAULT, PROP_NEPOCHS, config, res, int)

        if PROP_OPTIMIZER_LEARNINGRATE in config[SEC_TRAINER_DEFAULT]:
            _set_property(
                SEC_TRAINER_DEFAULT,
                PROP_OPTIMIZER_LEARNINGRATE,
                config,
                res,
                float
            )

        if PROP_OPTIMIZER_BETA1 in config[SEC_TRAINER_DEFAULT]:
            _set_property(
                SEC_TRAINER_DEFAULT,
                PROP_OPTIMIZER_BETA1,
                config,
                res,
                float
            )

        if PROP_OPTIMIZER_BETA2 in config[SEC_TRAINER_DEFAULT]:
            _set_property(
                SEC_TRAINER_DEFAULT,
                PROP_OPTIMIZER_BETA2,
                config,
                res,
                float
            )

        if PROP_OPTIMIZER_EPSILON in config[SEC_TRAINER_DEFAULT]:
            _set_property(
                SEC_TRAINER_DEFAULT,
                PROP_OPTIMIZER_EPSILON,
                config,
                res,
                float
            )

        if PROP_FCLUNITS in config[SEC_TRAINER_DEFAULT]:
            _set_property(SEC_TRAINER_DEFAULT, PROP_FCLUNITS, config, res, int)

        if PROP_DROPOUTLAYER_RATE in config[SEC_TRAINER_DEFAULT]:
            _set_property(
                SEC_TRAINER_DEFAULT,
                PROP_DROPOUTLAYER_RATE,
                config,
                res,
                float
            )

    # SECTION 4: Early stop trainer
    if _section_exists(SEC_TRAINER_EARLYSTOP, config):
        if PROP_MUWEIGHT in config[SEC_TRAINER_EARLYSTOP]:
            _set_property(
                SEC_TRAINER_EARLYSTOP,
                PROP_MUWEIGHT,
                config,
                res,
                float
            )

        if PROP_RHOWEIGHT in config[SEC_TRAINER_EARLYSTOP]:
            _set_property(
                SEC_TRAINER_EARLYSTOP,
                PROP_RHOWEIGHT,
                config,
                res,
                float
            )

    # SECTION 5: Trainer's tensorflow parameters
    if _section_exists(SEC_TRAINER_TENSORFLOW, config):
        if PROP_ENABLE_DEVICEPLACEMENT in config[SEC_TRAINER_TENSORFLOW]:
            _set_property(
                SEC_TRAINER_TENSORFLOW,
                PROP_ENABLE_DEVICEPLACEMENT,
                config,
                res,
                bool
            )

        if PROP_ENABLE_DISTRIBUTED in config[SEC_TRAINER_TENSORFLOW]:
            _set_property(
                SEC_TRAINER_TENSORFLOW,
                PROP_ENABLE_DISTRIBUTED,
                config,
                res,
                bool
            )

        if PROP_ALLOW_MEMORYGROWTH in config[SEC_TRAINER_TENSORFLOW]:
            _set_property(
                SEC_TRAINER_TENSORFLOW,
                PROP_ALLOW_MEMORYGROWTH,
                config,
                res,
                bool
            )

    # SECTION 6: Metadataset
    if _section_exists(SEC_METADATASET, config):
        if PROP_DATASETID in config[SEC_METADATASET]:
            _set_property(
                SEC_METADATASET,
                PROP_DATASETID,
                config,
                res,
                None
            )
        if PROP_TRAIN_TEST_SPLIT_PROP in config[SEC_METADATASET]:
            _set_property(
                SEC_METADATASET,
                PROP_TRAIN_TEST_SPLIT_PROP,
                config,
                res,
                float
            )
        if PROP_TFRECORDS_ROOTDIR in config[SEC_METADATASET]:
            _set_property(
                SEC_METADATASET,
                PROP_TFRECORDS_ROOTDIR,
                config,
                res,
                None
            )
        if PROP_RANDOMSEED in config[SEC_METADATASET]:
            _set_property(
                SEC_METADATASET,
                PROP_RANDOMSEED,
                config,
                res,
                int
            )
    return res


def _set_property(sec, prop, config, res, value_type=None):
    res[sec][prop] = _get_property(sec, prop, value_type, config)


def _section_exists(section_name, config):
    return section_name in config


def _property_exists(section_name, property_name, config):
    try:
        _ = config[section_name][property_name]
        return True
    except KeyError:
        return False


def _get_property(section_name, property_name, value_type, config):
    try:
        if value_type == bool:
            return config[section_name].getboolean(property_name)
        if value_type == int:
            return config[section_name].getint(property_name)
        if value_type == float:
            return config[section_name].getfloat(property_name)
        return config[section_name][property_name]
    except KeyError:
        return False
