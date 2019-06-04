"""Simple factories for default nas environment."""

import tensorflow as tf
from nasgym import nas_logger as logger
from nasgym import CONFIG_INI
import nasgym.utl.configreader as cr
import nasgym.envs.envspecs_parsers as parsers
from nasgym.dataset_handlers.default_handler import DefaultDatasetHandler
from nasgym.dataset_handlers.metadataset_handler import MetaDatasetHandler
from nasgym.net_ops.net_trainer import EarlyStopNASTrainer
from nasgym.net_ops.net_trainer import DefaultNASTrainer


class EnvSpecsParserFactory:
    """Factory of parsers."""

    @staticmethod
    def get_parser(config_file, parser_type):
        """Return the parser of interest."""
        if parser_type == "default":
            return parsers.DefaultEnvSpecsParser(config_file)
        if parser_type == "chained":
            # TODO: Make parser for chained structures
            pass
        raise ValueError("Unkown parser_type '{t}'".format(t=parser_type))

    @staticmethod
    def get_default_parser(config_file):
        """Return the default parser."""
        return EnvSpecsParserFactory.get_parser(config_file, "default")


class DatasetHandlerFactory:
    """Factory of parsers."""

    @staticmethod
    def get_handler(handler_type, **kwargs):
        """Return the parser of interest."""
        # a) Default handler (cifar10 from TensorFlow datasets)
        if handler_type == "default":
            (train_data, train_labels), (eval_data, eval_labels) = \
                tf.keras.datasets.cifar10.load_data()

            return DefaultDatasetHandler(
                train_data, train_labels, eval_data, eval_labels, "cifar10"
            )
        # b) Handler for Meta-dataset
        if handler_type == "meta-dataset":
            tfrecords_root, batch_size, split_prop, random_seed = \
                DatasetHandlerFactory._load_meta_dataset_attributes(**kwargs)

            return MetaDatasetHandler(
                tfrecords_rootdir=tfrecords_root,
                name="metadataset_handler",
                batch_size=batch_size,
                split_prop=split_prop,
                random_seed=random_seed
            )
        raise ValueError("Unkown handler_type '{t}'".format(t=handler_type))

    @staticmethod
    def get_default_handler():
        """Return the default parser."""
        return DatasetHandlerFactory.get_handler("default")

    @staticmethod
    def _load_meta_dataset_attributes(**kwargs):
        # Load the tfrecords_root
        try:
            tfrecords_root = \
                CONFIG_INI[cr.SEC_METADATASET][cr.PROP_TFRECORDS_ROOTDIR]
            logger.debug(
                "Reading tfrecords_root for meta-dataset from config.ini"
            )
        except KeyError:
            if 'tfrecords_root' in kwargs.keys():
                tfrecords_root = kwargs['tfrecords_root']
            else:
                raise ValueError(
                    "When using meta-dataset handler the tfrecords_root \
argument is expected."
                )
        finally:
            logger.debug("tfrecords_root set to: %s", tfrecords_root)

        # Load the batch size
        try:
            batch_size = CONFIG_INI[cr.SEC_TRAINER_DEFAULT][cr.PROP_BATCHSIZE]
            logger.debug("Reading batch size for meta-dataset from config.ini")
        except KeyError:
            batch_size = 256
        finally:
            logger.debug(
                "Batch size for meta-dataset handler set to: %d", batch_size
            )

        # Load the train/test split proportion
        try:
            split_prop = \
                CONFIG_INI[cr.SEC_METADATASET][cr.PROP_TRAIN_TEST_SPLIT_PROP]
            logger.debug(
                "Reading split proportion for meta-dataset from config.ini"
            )
        except KeyError:
            split_prop = 0.33
        finally:
            logger.debug(
                "Train-test split proportion for meta-dataset handler set to: \
%f", split_prop
            )

        # Load the random seed used for splitting.
        try:
            random_seed = \
                CONFIG_INI[cr.SEC_METADATASET][cr.PROP_RANDOMSEED]
            logger.debug("Reading random seed meta-dataset from config.ini")
        except KeyError:
            random_seed = 32
        finally:
            logger.debug(
                "Random seed for meta-dataset handler set to: %d", random_seed
            )
        return tfrecords_root, batch_size, split_prop, random_seed


class TrainerFactory:
    """Factory for the trainer class."""

    @staticmethod
    def get_trainer(trainer_type, state, dataset_handler, log_path,
                    variable_scope):
        if trainer_type == "default":
            batch_size = TrainerFactory._load_default_trainer_attributes()
            return DefaultNASTrainer(
                encoded_network=state,
                input_shape=dataset_handler.current_shape(),
                n_classes=dataset_handler.current_n_classes(),
                batch_size=batch_size,
                log_path=log_path,
                variable_scope=variable_scope
            )
        if trainer_type == "early-stop":
            batch_size = TrainerFactory._load_default_trainer_attributes()
            # pylint: disable=invalid-name
            rho, mu = TrainerFactory._load_early_stop_trainer_attributes()
            return EarlyStopNASTrainer(
                encoded_network=state,
                input_shape=dataset_handler.current_shape(),
                n_classes=dataset_handler.current_n_classes(),
                batch_size=batch_size,
                log_path=log_path,
                mu=mu,
                rho=rho,
                variable_scope=variable_scope
            )
        raise ValueError("Unkown trainer_type '{t}'".format(t=trainer_type))

    @staticmethod
    def get_default_trainer(state, dataset_handler, log_path, variable_scope):
        return TrainerFactory.get_trainer(
            "default", state, dataset_handler, log_path, variable_scope
        )

    # pylint: disable=no-method-argument
    def _load_default_trainer_attributes():
        # Load Batch size
        try:
            batch_size = CONFIG_INI[cr.SEC_TRAINER_DEFAULT][cr.PROP_BATCHSIZE]
            logger.debug("Reading batch size for trainer from config.ini")
        except KeyError:
            batch_size = 256
        finally:
            logger.debug("Batch size for trainer set to: %d", batch_size)

        return batch_size

    # pylint: disable=no-method-argument
    def _load_early_stop_trainer_attributes():
        try:
            rho = CONFIG_INI[cr.SEC_TRAINER_EARLYSTOP][cr.PROP_RHOWEIGHT]
            logger.debug("Reading rho for trainer from config.ini")
        except KeyError:
            rho = 0.5
        finally:
            logger.debug("Rho for trainer set to: %f", rho)

        try:
            mu = CONFIG_INI[cr.SEC_TRAINER_EARLYSTOP][cr.PROP_MUWEIGHT]
            logger.debug("Reading mu for trainer from config.ini")
        except KeyError:
            mu = 0.5
        finally:
            logger.debug("Mu for trainer set to: %f", mu)

        return rho, mu
