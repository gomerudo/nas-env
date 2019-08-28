"""Classes and methods for Net building."""

import math
import tensorflow as tf
from tensorflow.python.client import device_lib
import nasgym.utl.configreader as cr
from nasgym import nas_logger
from nasgym import CONFIG_INI
from nasgym.net_ops import LTYPE_ADD
from nasgym.net_ops import LTYPE_AVGPOOLING
from nasgym.net_ops import LTYPE_CONCAT
from nasgym.net_ops import LTYPE_CONVULUTION
from nasgym.net_ops import LTYPE_IDENTITY
from nasgym.net_ops import LTYPE_MAXPOOLING
from nasgym.net_ops import LTYPE_TERMINAL
from nasgym.net_ops.net_trainer import NasEnvTrainerBase
from nasgym.net_ops.net_builder import sequence_to_net


def sequence_to_net(sequence, input_tensor):
    """Build a network with TensorFlow, given a sequence of NSC."""
    # We use this list to store the built layers. Remember that each time we
    # iterate, we find the predecesor of the current layer, hence, the inputs
    # are never a problem. If for some reason the layer has no input, we should
    # throw an exception and ignore the layer
    tf_layers = {
        0: input_tensor
    }
    current_layer = None
    
    # We use this list to see if all layers were used to construct the graph
    non_used_layers = list(set([layer[0] for layer in sequence]))
    convolutions_count = 0
    for layer_encoding in sequence:
        layer_index = layer_encoding[0]
        layer_type = layer_encoding[1]
        layer_kernel_size = layer_encoding[2]
        layer_pred1 = layer_encoding[3]  # Predecesor 1
        layer_pred2 = layer_encoding[4]  # Predecesor 2

        if not layer_index:
            continue

        if layer_type == LTYPE_ADD:
            # i.e. if no predecesors at all
            if not layer_pred1 or not layer_pred2:
                raise ValueError(
                    "Invalid predecessors. Two predecessors are needed to \
build this network."
                )
                # logging.warning("No predecessors for layer %d", layer_index)

            with tf.name_scope("L{i}_ADD".format(i=layer_index)):
                current_layer = safe_add(
                    tensor_a=tf_layers[layer_pred1],
                    tensor_b=tf_layers[layer_pred2],
                    name="SafeAdd"
                )

        if layer_type == LTYPE_AVGPOOLING:
            with tf.name_scope("L{i}_AVGPOOL".format(i=layer_index)):
                current_layer = tf.keras.layers.AveragePooling2D(
                    pool_size=(layer_kernel_size, layer_kernel_size),
                    name="AvgPooling"
                )(tf_layers[layer_pred1])

        if layer_type == LTYPE_CONCAT:
            # i.e. if no predecesors at all
            if not layer_pred1 or not layer_pred2:
                # logging.warning("No predecessors for layer %d", layer_index)
                raise ValueError(
                    "Invalid predecessors. Two predecessors are needed to \
build this network."
                )

            with tf.name_scope("L{i}_CONCAT".format(i=layer_index)):
                current_layer = safe_concat(
                    tensor_a=tf_layers[layer_pred1],
                    tensor_b=tf_layers[layer_pred2],
                    name="SafeConcat"
                )

        if layer_type == LTYPE_CONVULUTION:
            with tf.name_scope("L{i}_PCC".format(i=layer_index)):
                relu_layer = tf.keras.layers.ReLU(
                    name="ReLU"
                )(tf_layers[layer_pred1])

                conv_layer = tf.keras.layers.Conv2D(
                    filters=32*(convolutions_count + 1),
                    kernel_size=(layer_kernel_size, layer_kernel_size),
                    # padding="same",
                    name="Conv"
                )(relu_layer)

                batch_norm = tf.keras.layers.BatchNormalization(
                    name="BatchNorm"
                )(conv_layer)

                current_layer = batch_norm
                convolutions_count += 1

        if layer_type == LTYPE_IDENTITY:
            with tf.name_scope("L{i}_IDENTITY".format(i=layer_index)):
                current_layer = tf.identity(
                    input=tf_layers[layer_pred1],
                    name="Identity"
                )

        if layer_type == LTYPE_MAXPOOLING:
            with tf.name_scope("L{i}_MAXPOOL".format(i=layer_index)):
                current_layer = tf.keras.layers.MaxPool2D(
                    pool_size=(layer_kernel_size, layer_kernel_size),
                    name="MaxPooling",
                    # padding="same"
                )(tf_layers[layer_pred1])

        if layer_type == LTYPE_TERMINAL:
            # We remove the terminal layer because we already visited it.
            try:
                non_used_layers.remove(layer_index)
            except ValueError:
                pass

            # Force the end of the building process. We ignore any remaining
            # portion of the sequence.
            break

        # Add the current layer to the dictionary
        tf_layers[layer_index] = current_layer

        # Mark as used:
        #   Two different try-except to always remove both of the predecesors
        try:
            non_used_layers.remove(layer_pred1)
        except ValueError:
            pass

        try:
            non_used_layers.remove(layer_pred2)
        except ValueError:
            pass

    # Handle the non used layers: concatenate one by one.
    #   The last layer before terminate is never used, but if there are two or
    #   more, then we need to concatenate one by one.
    if non_used_layers:
        # print("Non used", non_used_layers)
        pivot = tf_layers[non_used_layers[0]]
        for idx in range(1, len(non_used_layers)):
            with tf.name_scope("END_CONCAT{i}".format(i=idx)):
                pivot = safe_concat(
                    tensor_a=pivot,
                    tensor_b=tf_layers[non_used_layers[idx]],
                    name="SafeConcat"
                )
        current_layer = pivot

    # current_layer is the last used layer
    return current_layer


def safe_concat(tensor_a, tensor_b, name):
    """Concatenate two tensors even if they have different shapes.

    The fix of the shapes is done with a zero-padding on both tensors.
    """
    fixed_b = fix_tensor_shape(
        tensor_target=tensor_b,
        tensor_reference=tensor_a,
        free_axis=1
    )
    fixed_a = fix_tensor_shape(
        tensor_target=tensor_a,
        tensor_reference=tensor_b,
        free_axis=1
    )

    concatenated = tf.keras.layers.concatenate(
        inputs=[
            fixed_a,
            fixed_b,
        ],
        axis=3,
        name=name
    )

    return concatenated


def safe_add(tensor_a, tensor_b, name):
    """Concatenate two tensors even if they have different shapes.

    The fix of the shapes is done with a zero-padding on both tensors.
    """
    fixed_b = fix_tensor_shape(
        tensor_target=tensor_b,
        tensor_reference=tensor_a,
        free_axis=1
    )
    fixed_a = fix_tensor_shape(
        tensor_target=tensor_a,
        tensor_reference=tensor_b,
        free_axis=1
    )

    added = tf.keras.layers.add(
        inputs=[
            fixed_a,
            fixed_b,
        ],
        name=name
    )

    return added


def is_same_rank(tensor_a, tensor_b):
    """Verify whether the rank of two tensors is the same."""
    return tensor_a.get_shape().rank == tensor_b.get_shape().rank


def fix_tensor_shape(tensor_target, tensor_reference, free_axis=1, name="pad"):
    """Fix a tensor's shape with respect to a reference using padding."""
    ref_shape = tensor_reference.get_shape().dims
    target_shape = tensor_target.get_shape().dims
    target_rank = len(target_shape)
    ref_rank = len(ref_shape)

    if ref_rank != target_rank:
        raise ValueError("Tensors must have the same dimension.")

    if free_axis < 0:
        free_axis = ref_rank

    free_axis -= 1  # Shift the axis to start with 0 for simplicity

    paddings_arg = []
    for it_axis in range(target_rank):
        # If the current axis is the free axis then pad nothing, i.e. (rank, 0)
        if it_axis == free_axis:
            paddings_arg.append([0, 0])
            continue

        # Compute the difference of the axes to know how many pads are needed
        if ref_shape[it_axis].value is None or \
                target_shape[it_axis].value is None:
            axes_diff = 0
        else:
            axes_diff = ref_shape[it_axis].value - target_shape[it_axis].value

        # If target axis has a higher order than reference then do not pad.
        if axes_diff < 0:
            axes_diff = 0

        # If everything is ok, we simply store the desired fix-value
        paddings_arg.append([axes_diff//2, axes_diff - axes_diff//2])

    # print("Padding with list", paddings_arg)

    padded_tensor = tf.pad(
        tensor=tensor_target,
        paddings=tf.constant(paddings_arg),
        mode="CONSTANT",
        constant_values=0,
        name=name + "padding"
    )

    return padded_tensor


class NetEvaluation(NasEnvTrainerBase):

    def __init__(self, encoded_network, input_shape, n_classes, batch_size=256,
                 log_path="./trainer", variable_scope="evaluation",
                 n_epochs=12, op_beta1=0.9,
                 op_beta2=0.999, op_epsilon=10e-08, fcl_units=1024,
                 dropout_rate=0.4, n_obs_train=None):
        """Specific constructor with option for FLOPS and Density."""
        super(NetEvaluation, self).__init__(
            encoded_network=encoded_network,
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
        self.fcl_units = fcl_units
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
                        learning_rate = tf.train.cosine_decay(
                            learning_rate=0.1,
                            global_step=global_step,
                            decay_steps=self.n_epochs,
                            name="COS_DECAY"
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
