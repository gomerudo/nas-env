"""Module to implement training operations for Neural Networks."""

import os
import math
from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.python.client import device_lib
import nasgym.utl.configreader as cr
from nasgym import nas_logger
from nasgym import CONFIG_INI
from nasgym.net_ops.net_builder import sequence_to_net
from nasgym.net_ops.net_utils import compute_network_density
from nasgym.net_ops.net_utils import compute_network_flops


class NasEnvTrainerBase(ABC):
    """Simple trainer interface."""

    def __init__(self, encoded_network, input_shape, n_classes,
                 batch_size=256, log_path="./trainer",
                 variable_scope="custom", profile_path="./profiler"):
        """General purpose constructor."""
        # Encapsulation
        self.encoded_network = encoded_network
        self.input_shape = input_shape
        self.n_clases = n_classes
        self.batch_size = batch_size
        self.log_path = log_path
        self.variable_scope = variable_scope
        self.tf_partial_network = None
        self.classifier = None
        # Init of superclass
        super().__init__()

    @abstractmethod
    def build_model_fn(self):
        """Add the training graph."""

    @abstractmethod
    def train(self, train_data, train_labels, train_input_fn, n_epochs):
        """Abstract method to implement training."""
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    def evaluate(self, eval_data, eval_labels, eval_input_fn):
        """Abstract method to implement evaluation."""
        raise NotImplementedError("Method must be implemented by subclass")


class OomReportingHook(tf.train.SessionRunHook):
    """Report OOM during training when using Estimator."""

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(
            fetches=[],  # no extra fetches
            options=tf.RunOptions(
                report_tensor_allocations_upon_oom=True
            )
        )


class DefaultNASTrainer(NasEnvTrainerBase):
    """Implement Training with Eearly Stop Strategy."""

    def __init__(self, encoded_network, input_shape, n_classes, batch_size=256,
                 log_path="./trainer", variable_scope="custom",
                 profile_path="./profiler", op_decay_steps=12, op_beta1=0.9,
                 op_beta2=0.999, op_epsilon=10e-08, fcl_units=1024,
                 dropout_rate=0.4, n_obs_train=None):
        """Specific constructor with option for FLOPS and Density."""
        super(DefaultNASTrainer, self).__init__(
            encoded_network=encoded_network,
            input_shape=input_shape,
            n_classes=n_classes,  # TODO: We don't use it.
            batch_size=batch_size,
            log_path=log_path,
            variable_scope=variable_scope,
            profile_path=profile_path
        )
        self.op_decay_steps = op_decay_steps
        self.op_beta1 = op_beta1
        self.op_beta2 = op_beta2
        self.op_epsilon = op_epsilon
        self.fcl_units = fcl_units
        self.dropout_rate = dropout_rate
        self.n_obs_train = n_obs_train
        # Total number of steps
        self._n_steps = \
            self.op_decay_steps * math.ceil(self.n_obs_train/self.batch_size)
        self._n_steps = math.ceil(self._n_steps)
        # Steps per epoch
        self._steps_per_epoch = math.floor(self._n_steps/self.op_decay_steps)
        self._set_estimator()

    def _set_estimator(self):
        nas_logger.debug(
            "Configuring the estimator that will be used for training and \
evaluation"
        )
        if self.classifier is None:
            # Read the configuration for the distributed strategy (config.ini)
            try:
                aux = CONFIG_INI[cr.SEC_TRAINER_TENSORFLOW]
                self._distributed_enabled = aux[cr.PROP_ENABLE_DISTRIBUTED]
            except KeyError:
                self._distributed_enabled = False

            # Read the configuration for the log device placement (config.ini)
            try:
                aux = CONFIG_INI[cr.SEC_TRAINER_TENSORFLOW]
                self._devplacement_enabled = \
                    aux[cr.PROP_ENABLE_DEVICEPLACEMENT]
            except KeyError:
                self._devplacement_enabled = False

            # Read the configuration for the memory growth (config.ini)
            try:
                aux = CONFIG_INI[cr.SEC_TRAINER_TENSORFLOW]
                allow_memory_growth = aux[cr.PROP_ALLOW_MEMORYGROWTH]
            except KeyError:
                allow_memory_growth = False

            # Actually evaluation if distributed strategy is enabled
            if self._distributed_enabled:
                nas_logger.info("Distributed strategy has been enabled")
                # Obtain the available GPUs
                local_device = device_lib.list_local_devices()
                gpu_devices = \
                    [x.name for x in local_device if x.device_type == 'GPU']
                self.distributed_nreplicas = len(gpu_devices)

                distributed_strategy = tf.contrib.distribute.MirroredStrategy(
                    num_gpus=self.distributed_nreplicas
                )
                nas_logger.info(
                    "Number of replicas: %d", self.distributed_nreplicas
                )
            else:
                distributed_strategy = None
                self.distributed_nreplicas = 1

            # Evaluating if log device placement
            if self._devplacement_enabled:
                nas_logger.debug(
                    "Distributed strategy has been indicated. Obtaining the \
number of replicas available."
                )
                sess_config = tf.ConfigProto(log_device_placement=True)
            else:
                sess_config = tf.ConfigProto()

            # pylint: disable=no-member
            if allow_memory_growth:
                nas_logger.debug(
                    "Dynamic memory growth for TensorFlow is enabled"
                )
                sess_config.gpu_options.allow_growth = True

            # pylint: disable=no-member
            run_config = tf.estimator.RunConfig(
                session_config=sess_config,
                # save_checkpoints_steps=5,
                # save_checkpoints_secs=None,
                train_distribute=distributed_strategy,
                eval_distribute=distributed_strategy
            )
            # pylint: disable=no-member
            self.classifier = tf.estimator.Estimator(
                config=run_config,
                model_fn=self.build_model_fn(),
                model_dir="{root_dir}/{model_dir}".format(
                    root_dir=self.log_path,
                    model_dir="model"
                )
            )

    def build_model_fn(self):
        """Implement training of network with custom approach."""
        # Define the model_fn we want to return
        def model_fn(features, labels, mode):
            with tf.variable_scope(self.variable_scope):
                # 1. Define the input placeholder
                if len(self.input_shape) == 2:
                    nas_logger.debug("Reshaping input during model building.")
                    net_input = tf.reshape(
                        tensor=features["x"],
                        shape=[-1] + list(self.input_shape) + [1],
                        name="L0_RESHAPE"
                    )
                else:
                    net_input = features["x"]

                # 2. Simply call the network
                self.tf_partial_network = sequence_to_net(
                    sequence=self.encoded_network,
                    input_tensor=net_input
                )

                # 3. Build the Fully-Connected layers after block.
                with tf.name_scope("L_FC"):
                    # Flatten and connect to the Dense Layer
                    ll_flat = tf.layers.flatten(
                        inputs=self.tf_partial_network,
                        name="Flatten"
                    )
                    dense_layer = tf.layers.dense(
                        inputs=ll_flat,
                        units=self.fcl_units,
                        activation=tf.nn.relu,
                        name="DENSE"
                    )
                    dropout_layer = tf.layers.dropout(
                        inputs=dense_layer,
                        rate=self.dropout_rate,
                        # pylint: disable=no-member
                        training=mode == tf.estimator.ModeKeys.TRAIN,
                        name="DROPOUT"
                    )

                # 4. Build the Prediction Layer based on a Softmax
                with tf.name_scope("L_PRED"):
                    # Logits layer
                    logits_layer = tf.layers.dense(
                        inputs=dropout_layer,
                        units=self.n_clases,
                        name="PL_Logits"
                    )

                    predictions = {
                        "classes": tf.argmax(
                            input=logits_layer,
                            axis=1,
                            name="PL_Classes"
                        ),
                        "probabilities": tf.nn.softmax(
                            logits=logits_layer,
                            name="PL_Softmax"
                        )
                    }

                    # If we are asked for prediction only, we return the
                    # prediction and stop adding nodes to the graph.
                    # pylint: disable=no-member
                    if mode == tf.estimator.ModeKeys.PREDICT:
                        return tf.estimator.EstimatorSpec(
                            mode=mode,
                            predictions=predictions
                        )

                # 4. Build the training nodes
                with tf.name_scope("L_TRAIN"):
                    # Loss
                    loss_layer = tf.losses.sparse_softmax_cross_entropy(
                        labels=labels,
                        logits=logits_layer
                    )

                    # Training Op
                    # pylint: disable=no-member
                    if mode == tf.estimator.ModeKeys.TRAIN:
                        # The optimizer via Gradient Descent (we can change it)

                        global_step = tf.train.get_global_step()
                        # learning_rate = tf.train.exponential_decay(
                        #     learning_rate=0.0001,
                        #     global_step=global_step,
                        #     decay_steps=self.op_decay_steps,
                        #     decay_rate=0.02
                        # )

                        # The paper's version of the learning rate
                        n_reductions = math.floor(self.op_decay_steps/5)
                        learning_rate = 0.001
                        ul = self.op_decay_steps*self._steps_per_epoch
                        for i in range(1, n_reductions + 1):
                            ll = self._steps_per_epoch*5*(i-1) + 1
                            if global_step in range(ll, ul+1) and i > 0:
                                learning_rate *= 0.2

                        optimizer = tf.train.AdamOptimizer(
                            learning_rate=learning_rate,
                            beta1=self.op_beta1,
                            beta2=self.op_beta2,
                            epsilon=self.op_epsilon,
                            name="OPT"
                        )
                        # We say that we want to optimize the loss layer using
                        # the optimizer.
                        train_op = optimizer.minimize(
                            loss=loss_layer,
                            global_step=global_step,
                            name="OPT_MIN"
                        )
                        # And return
                        # pylint: disable=no-member
                        return tf.estimator.EstimatorSpec(
                            mode=mode,
                            loss=loss_layer,
                            train_op=train_op
                        )

                # 5. Build the evaluation nodes.
                with tf.name_scope("L_EVAL"):
                    # Evaluation metric is accuracy
                    eval_metric_ops = {
                        "accuracy": tf.metrics.accuracy(
                            labels=labels,
                            predictions=predictions["classes"],
                            name="ACC"
                        )
                    }

                    # pylint: disable=no-member
                    return tf.estimator.EstimatorSpec(
                        mode=mode,
                        loss=loss_layer,
                        eval_metric_ops=eval_metric_ops
                    )
            # End of tf.variable_scope()

        # Return the model_fn function
        return model_fn

    def train(self, train_data, train_labels, train_input_fn="default",
              n_epochs=12, n_obs=0):
        """Train the self-network with the the given training configuration."""
        if isinstance(train_input_fn, str):
            if train_input_fn == "default":
                # pylint: disable=no-member
                train_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": train_data},
                    y=train_labels,
                    batch_size=self.batch_size,
                    num_epochs=None,
                    shuffle=True
                )
            else:
                raise ValueError(
                    "train_input_fn has been specified as string, but no \
valid value has been provided. Options are: 'default'"
                )

        nas_logger.debug("Running tensorflow training for %d epochs", n_epochs)

        steps = n_epochs * math.ceil(self.n_obs_train/self.batch_size)
        nas_logger.debug(
            "Running tensorflow training for %d epochs (%d steps)",
            n_epochs,
            steps
        )
        train_res = self.classifier.train(
            input_fn=train_input_fn,
            steps=steps,
        )
        nas_logger.debug("TensorFlow training finished")

        return train_res

    def evaluate(self, eval_data, eval_labels, eval_input_fn="default"):
        """Evaluate a given dataset, with the internal network."""
        # Validations:
        # If it is of type str, make sure is a valid
        if isinstance(eval_input_fn, str):
            if eval_input_fn == "default":
                # pylint: disable=no-member
                eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": eval_data},
                    y=eval_labels,
                    num_epochs=1,
                    # batch_size=self.batch_size,
                    shuffle=False
                )

        nas_logger.debug("Running tensorflow evaluation")
        eval_res = self.classifier.evaluate(input_fn=eval_input_fn)
        nas_logger.debug("TensorFlow evaluation finished")
        return eval_res


class EarlyStopNASTrainer(DefaultNASTrainer):
    """Implement Training with Eearly Stop Strategy."""

    def __init__(self, encoded_network, input_shape, n_classes, batch_size=256,
                 log_path="./trainer", mu=0.5, rho=0.5, variable_scope="cnn",
                 profile_path="./profiler", op_decay_steps=12, op_beta1=0.9,
                 op_beta2=0.999, op_epsilon=10e-08, fcl_units=1024,
                 dropout_rate=0.4, n_obs_train=None):
        """Specific constructor with option for FLOPS and Density."""
        super(EarlyStopNASTrainer, self).__init__(
            encoded_network=encoded_network,
            input_shape=input_shape,
            n_classes=n_classes,
            batch_size=batch_size,
            log_path=log_path,
            variable_scope=variable_scope,
            profile_path=profile_path,
            op_decay_steps=op_decay_steps,
            op_beta1=op_beta1,
            op_beta2=op_beta2,
            op_epsilon=op_epsilon,
            fcl_units=fcl_units,
            dropout_rate=dropout_rate,
            n_obs_train=n_obs_train
        )
        # Custom variables for the refined accuracy in BlockQNN implementation
        # pylint: disable=invalid-name
        self.mu = mu
        self.rho = rho

        # Updated during training call
        self.density = None
        self.flops = None

        # Build the estimator
        self._set_estimator()

    def build_model_fn(self):
        """Implement training of network with custom approach."""
        # Define the model_fn we want to return
        def model_fn(features, labels, mode):
            with tf.variable_scope(self.variable_scope):
                # 1. Define the input placeholder
                if len(self.input_shape) == 2:  # Reshape if necessary
                    new_shape = [-1] + list(self.input_shape) + [1]
                    net_input = tf.reshape(
                        tensor=features["x"],
                        shape=new_shape,
                        name="L0_RESHAPE"
                    )
                else:
                    net_input = features["x"]

                # 2. Simply call the network
                self.tf_partial_network = sequence_to_net(
                    sequence=self.encoded_network,
                    input_tensor=net_input
                )

                # 3. Call here the functions for flops & density to avoid more
                # elements. The check is done because for some reason, the
                # number of FLOPS changes during training.
                if self.flops is None:
                    self.flops = compute_network_flops(
                        graph=tf.get_default_graph(),
                        collection_name=self.variable_scope,
                        logdir=self.log_path
                    )

                if self.density is None:
                    self.density = compute_network_density(
                        graph=tf.get_default_graph(),
                        collection_name=self.variable_scope
                    )

                # 4. Build the fully-connected layer after the block
                with tf.name_scope("L_FC"):
                    # Flatten and connect to the Dense Layer
                    ll_flat = tf.layers.flatten(
                        inputs=self.tf_partial_network,
                        name="Flatten"
                    )
                    dense_layer = tf.layers.dense(
                        inputs=ll_flat,
                        units=self.fcl_units,
                        activation=tf.nn.relu,
                        name="DENSE"
                    )
                    dropout_layer = tf.layers.dropout(
                        inputs=dense_layer,
                        rate=self.dropout_rate,
                        # pylint: disable=no-member
                        training=mode == tf.estimator.ModeKeys.TRAIN,
                        name="DROPOUT"
                    )

                # 5. Build the prediction layer, based on a softmax
                with tf.name_scope("L_PRED"):
                    # Logits layer
                    logits_layer = tf.layers.dense(
                        inputs=dropout_layer,
                        units=self.n_clases,
                        name="PL_Logits"
                    )

                    predictions = {
                        "classes": tf.argmax(
                            input=logits_layer,
                            axis=1,
                            name="PL_Classes"
                        ),
                        "probabilities": tf.nn.softmax(
                            logits=logits_layer,
                            name="PL_Softmax"
                        )
                    }

                    # If we are asked for prediction only, we return the
                    # prediction and stop adding nodes to the graph.
                    # pylint: disable=no-member
                    if mode == tf.estimator.ModeKeys.PREDICT:
                        return tf.estimator.EstimatorSpec(
                            mode=mode,
                            predictions=predictions
                        )

                # Build the training nodes
                with tf.name_scope("L_TRAIN"):
                    # Loss
                    loss_layer = tf.losses.sparse_softmax_cross_entropy(
                        labels=labels,
                        logits=logits_layer
                    )

                    # Training Op
                    # pylint: disable=no-member
                    if mode == tf.estimator.ModeKeys.TRAIN:
                        # The optimizer via Gradient Descent (we can change it)
                        global_step = tf.train.get_global_step()
                        n_reductions = math.floor(self.op_decay_steps/5)
                        learning_rate = 0.001
                        ul = self.op_decay_steps*self._steps_per_epoch
                        for i in range(1, n_reductions + 1):
                            ll = self._steps_per_epoch*5*(i-1) + 1
                            if global_step in range(ll, ul+1) and i > 0:
                                learning_rate *= 0.2

                        optimizer = tf.train.AdamOptimizer(
                            learning_rate=learning_rate,
                            beta1=self.op_beta1,
                            beta2=self.op_beta2,
                            epsilon=self.op_epsilon,
                            name="OPT"
                        )
                        # We say that we want to optimize the loss layer using
                        # the optimizer.
                        train_op = optimizer.minimize(
                            loss=loss_layer,
                            global_step=global_step,
                            name="OPT_MIN"
                        )
                        # And return
                        # pylint: disable=no-member
                        return tf.estimator.EstimatorSpec(
                            mode=mode,
                            loss=loss_layer,
                            train_op=train_op
                        )

                # Build the evaluation nodes (regular accuracy).
                with tf.name_scope("L_EVAL"):
                    # Evaluation metric is accuracy
                    eval_metric_ops = {
                        "accuracy": tf.metrics.accuracy(
                            labels=labels,
                            predictions=predictions["classes"],
                            name="ACC"
                        )
                    }

                    # pylint: disable=no-member
                    return tf.estimator.EstimatorSpec(
                        mode=mode,
                        loss=loss_layer,
                        eval_metric_ops=eval_metric_ops
                    )

        # Return the model_fn function
        return model_fn

    @property
    def weighted_log_density(self):
        """Return the weighted version of the logarithm of the density."""
        try:
            return self.rho*math.log(self.density)
        except ValueError:
            return 0

    @property
    def weighted_log_flops(self):
        """Return the weighted version of the logarithm of the FLOPs."""
        try:
            return self.mu*math.log(self.flops)
        except ValueError:
            return 0
