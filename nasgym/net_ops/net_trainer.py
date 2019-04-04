"""Module to implement training operations for Neural Networks."""

from abc import ABC, abstractmethod
import tensorflow as tf
from nas_gym.net_ops.net_builder import sequence_to_net


class NasEnvTrainerBase(ABC):
    """Simple trainer class."""

    def __init__(self, encoded_network, input_shape, n_classes,
                 batch_size=256, log_path="./trainer"):
        """General purpose constructor."""
        # Encapsulation
        self.encoded_network = encoded_network
        self.input_shape = input_shape
        self.n_clases = n_classes
        self.batch_size = batch_size
        self.log_path = log_path
        self.tf_partial_network = None
        self.classifier = None
        # Init of superclass
        super().__init__()

    @abstractmethod
    def build_model_fn(self):
        """Add the training graph."""

    @abstractmethod
    def train(self, train_input_fn, n_epochs):
        """Abstract method to implement training."""
        raise NotImplementedError("Method must be implemented by subclass")

    def evaluate(self):
        """Abstract method to implement evaluation."""
        raise NotImplementedError("Method must be implemented by subclass")


class DefaultNASTrainer(NasEnvTrainerBase):
    """Implement Training with Eearly Stop Strategy."""

    def __init__(self, network, input_shape, n_classes, batch_size=256,
                 log_path="./trainer"):
        """Specific constructor with option for FLOPS and Density."""
        super(DefaultNASTrainer, self).__init__(
            encoded_network=network,
            input_shape=input_shape,
            n_classes=n_classes,
            batch_size=batch_size,
            log_path=log_path
        )
        self._set_estimator()

    def _set_estimator(self):
        if self.classifier is None:
            # pylint: disable=no-member
            self.classifier = tf.estimator.Estimator(
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
            # 1. Define the input placeholder
            if len(self.input_shape) == 2:
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

            with tf.name_scope("L_FC"):
                # Flatten and connect to the Dense Layer
                ll_flat = tf.layers.flatten(
                    inputs=self.tf_partial_network,
                    name="Flatten"
                )
                dense_layer = tf.layers.dense(
                    inputs=ll_flat,
                    units=1024,
                    activation=tf.nn.relu,
                    name="DENSE"
                )
                dropout_layer = tf.layers.dropout(
                    inputs=dense_layer,
                    rate=0.4,
                    # pylint: disable=no-member
                    training=mode == tf.estimator.ModeKeys.TRAIN,
                    name="DROPOUT"
                )

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

                # If we are asked for prediction only, we return the prediction
                # and stop adding nodes to the graph.
                # pylint: disable=no-member
                if mode == tf.estimator.ModeKeys.PREDICT:
                    return tf.estimator.EstimatorSpec(
                        mode=mode,
                        predictions=predictions
                    )

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
                    optimizer = tf.train.AdamOptimizer(
                        learning_rate=0.001,
                        beta1=0.9,
                        beta2=0.999,
                        epsilon=10e-08,
                        name="OPT"
                    )
                    # We say that we want to optimize the loss layer using the
                    # optimizer.
                    train_op = optimizer.minimize(
                        loss=loss_layer,
                        global_step=tf.train.get_global_step(),
                        name="OPT_MIN"
                    )
                    # And return
                    # pylint: disable=no-member
                    return tf.estimator.EstimatorSpec(
                        mode=mode,
                        loss=loss_layer,
                        train_op=train_op
                    )

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

    def train(self, train_data, train_labels, train_input_fn="default",
              n_epochs=12):
        # Validations:
        # If it is of type str, make sure is a valid
        if isinstance(train_input_fn, str):
            # We use a list in case we want to extend in the future.
            if train_input_fn in ["default"]:
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
                    "train_input_fn has been specified as string, but no valid \
value has been provided. Options are: 'default'"
                )

        # Prepare for logging of the probabilities, i.e. the softmax layer
        tensors_to_log = {"probabilities": "L_PRED/PL_Softmax"}

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log,
            every_n_iter=1
        )

        train_res = self.classifier.train(
            input_fn=train_input_fn,
            steps=n_epochs,
            hooks=[logging_hook]
        )

        return train_res

    def evaluate(self, eval_data, eval_labels, eval_input_fn="default"):
        # Validations:
        # If it is of type str, make sure is a valid
        if isinstance(eval_input_fn, str):
            # We use a list in case we want to extend in the future.
            if eval_input_fn in ["default"]:
                if eval_input_fn == "default":
                    # pylint: disable=no-member
                    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                        x={"x": eval_data},
                        y=eval_labels,
                        num_epochs=1,
                        shuffle=False
                    )

        eval_res = self.classifier.evaluate(input_fn=eval_input_fn)
        return eval_res


# class TrainerEarlyStop(TrainerBase):
#     """Implement Training with Eearly Stop Strategy."""

#     def __init__(self, network, log_path, refine=True, mu=0.5, rho=0.5):
#         """Specific constructor with option for FLOPS and Density."""
#         self.refine = refine
#         self.mu = mu
#         self.rho = rho
#         super.__init__(network, log_path)

#     def add_train_graph(self, target_input):
#         """Implement training of network with custom approach."""