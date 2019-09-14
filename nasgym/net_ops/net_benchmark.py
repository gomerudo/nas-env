"""Classes and methods for Net building."""

import math
import tensorflow as tf
import nasgym.utl.configreader as cr
from nasgym import nas_logger
from nasgym import CONFIG_INI
from nasgym.net_ops.net_trainer import NasEnvTrainerBase


def vgg_net_builder(input_tensor):
    """Build a network with TensorFlow, given a sequence of NSC."""
    # We use this list to store the built layers. Remember that each time we
    # iterate, we find the predecesor of the current layer, hence, the inputs
    # are never a problem. If for some reason the layer has no input, we should
    # throw an exception and ignore the layer

    current_layer = input_tensor

    with tf.name_scope("Block1"):
        # Conv 1
        current_layer = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            name="Conv1"
        )(current_layer)

        current_layer = tf.keras.layers.ReLU(
            name="ReLU"
        )(current_layer)

        # Conv 2
        current_layer = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            name="Conv2"
        )(current_layer)

        current_layer = tf.keras.layers.ReLU(
            name="ReLU"
        )(current_layer)

        # Pooling
        current_layer = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            name="MaxPooling",
            # padding="same"
        )(current_layer)

    with tf.name_scope("Block2"):
        # Conv 1
        current_layer = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            name="Conv1"
        )(current_layer)

        current_layer = tf.keras.layers.ReLU(
            name="ReLU"
        )(current_layer)

        # Conv 2
        current_layer = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            name="Conv2"
        )(current_layer)

        current_layer = tf.keras.layers.ReLU(
            name="ReLU"
        )(current_layer)

        # Pooling
        current_layer = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            name="MaxPooling",
            # padding="same"
        )(current_layer)

    # with tf.name_scope("Block3"):
    #     # Conv 1
    #     current_layer = tf.keras.layers.Conv2D(
    #         filters=256,
    #         kernel_size=(3, 3),
    #         padding="same",
    #         name="Conv1"
    #     )(current_layer)

    #     current_layer = tf.keras.layers.ReLU(
    #         name="ReLU"
    #     )(current_layer)

    #     # Conv 2
    #     current_layer = tf.keras.layers.Conv2D(
    #         filters=256,
    #         kernel_size=(3, 3),
    #         padding="same",
    #         name="Conv2"
    #     )(current_layer)

    #     current_layer = tf.keras.layers.ReLU(
    #         name="ReLU"
    #     )(current_layer)

    #     # Pooling
    #     current_layer = tf.keras.layers.MaxPool2D(
    #         pool_size=(2, 2),
    #         strides=(2, 2),
    #         name="MaxPooling",
    #         # padding="same"
    #     )(current_layer)

    # with tf.name_scope("Block4"):
    #     # Conv 1
    #     current_layer = tf.keras.layers.Conv2D(
    #         filters=512,
    #         kernel_size=(3, 3),
    #         padding="same",
    #         name="Conv1"
    #     )(current_layer)

    #     current_layer = tf.keras.layers.ReLU(
    #         name="ReLU"
    #     )(current_layer)

    #     # Conv 2
    #     current_layer = tf.keras.layers.Conv2D(
    #         filters=512,
    #         kernel_size=(3, 3),
    #         padding="same",
    #         name="Conv2"
    #     )(current_layer)

    #     current_layer = tf.keras.layers.ReLU(
    #         name="ReLU"
    #     )(current_layer)

    #     # Pooling
    #     current_layer = tf.keras.layers.MaxPool2D(
    #         pool_size=(2, 2),
    #         strides=(2, 2),
    #         name="MaxPooling",
    #         # padding="same"
    #     )(current_layer)

    with tf.name_scope("DenseBlock"):
        current_layer = tf.layers.flatten(
            inputs=current_layer,
            name="Flatten"
        )
        current_layer = tf.layers.dense(
            inputs=current_layer,
            units=4096,
            activation=tf.nn.relu,
            name="Dense1"
        )
        current_layer = tf.layers.dense(
            inputs=current_layer,
            units=4096,
            activation=tf.nn.relu,
            name="Dense2"
        )
    return current_layer


class NetBenchmarking(NasEnvTrainerBase):

    def __init__(self, input_shape, n_classes, batch_size=256,
                 log_path="./trainer", variable_scope="evaluation",
                 n_epochs=12, op_beta1=0.9,
                 op_beta2=0.999, op_epsilon=10e-08,
                 dropout_rate=0.4, n_obs_train=None):
        """Specific constructor with option for FLOPS and Density."""
        super(NetBenchmarking, self).__init__(
            encoded_network=None,
            input_shape=input_shape,
            n_classes=n_classes,  # TODO: We don't use it.
            batch_size=batch_size,
            log_path=log_path,
            variable_scope=variable_scope,
        )
        self.n_epochs = n_epochs
        self.op_beta1 = op_beta1
        self.op_beta2 = op_beta2
        self.op_epsilon = op_epsilon
        self.dropout_rate = dropout_rate
        self.n_obs_train = n_obs_train
        # Total number of steps
        self._n_steps = \
            self.n_epochs * math.ceil(self.n_obs_train/self.batch_size)
        self._n_steps = math.ceil(self._n_steps)
        # Steps per epoch
        self._steps_per_epoch = math.floor(self._n_steps/self.n_epochs)
        self._set_estimator()
        # An empty list storting all accuracies found during evaluation
        self.eval_accuracies = []

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
                save_checkpoints_steps=10,
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
                self.tf_partial_network = vgg_net_builder(
                    input_tensor=net_input
                )

                # 3. Build the trainer

                # 4. Build the Prediction Layer based on a Softmax
                with tf.name_scope("L_PRED"):
                    # Logits layer
                    logits_layer = tf.layers.dense(
                        inputs=self.tf_partial_network,
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
                        learning_rate = tf.train.exponential_decay(
                            learning_rate=0.001,
                            global_step=global_step,
                            decay_steps=self.n_epochs,
                            decay_rate=0.90,
                            name="DECAY"
                        )

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
                    spec = tf.estimator.EstimatorSpec(
                        mode=mode,
                        loss=loss_layer,
                        eval_metric_ops=eval_metric_ops
                    )
                    return spec
            # End of tf.variable_scope()

        # Return the model_fn function
        return model_fn

    def train(self, train_data, train_labels, train_input_fn="default",
              n_epochs=12):
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
                    shuffle=False
                )

        nas_logger.debug("Running tensorflow evaluation")
        eval_res = self.classifier.evaluate(input_fn=eval_input_fn)
        nas_logger.debug("TensorFlow evaluation finished")
        return eval_res
