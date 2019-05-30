"""Metadataset handler."""

import math
import glob
import tensorflow as tf
from nasgym.dataset_handlers.default_handler import AbstractDatasetHandler
from nasgym import CONFIG_INI
from nasgym.utl import configreader as cr


def n_elements(records_list):
    """Return the number of elements in a tensorflow records file."""
    count = 0
    for tfrecords_file in records_list:
        for _ in tf.python_io.tf_record_iterator(tfrecords_file):
            count += 1
    return count


def parser(record_dataset):
    """Parse a given TFRecordsDataset object."""
    # This is the definition we expect in the TFRecords for meta-dataset
    features = {
        'image': tf.FixedLenFeature([], dtype=tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }
    exp_image_size = 84

    # 1. We parse the record_dataset with the features defined above.
    parsed = tf.parse_single_example(record_dataset, features)

    # 2. We will decode the image as a jpeg with 3 channels and resize it to
    #    the expected image size
    image_decoded = tf.image.decode_jpeg(parsed['image'], channels=3)
    image_resized = tf.image.resize_images(
        image_decoded,
        [exp_image_size, exp_image_size],
        method=tf.image.ResizeMethod.BILINEAR,
        align_corners=True
    )
    # 3. And we normalize the dataset in the range [0, 1]
    image_normalized = image_resized / 255.0

    # 4. we make the label an int32.
    label = tf.cast(parsed['label'], tf.int32)

    # 5. We return as dataset a s pair ( {features}, label)
    return {'x': image_normalized}, label


def metadataset_input_fn(tfrecord_data, data_length, batch_size=128, 
                         is_train=True, split_prop=0.33, random_seed=32):
    """Input function for a tensorflow estimator."""
    # 1. Compute the length of the train-validation split
    trainset_length = math.floor(data_length*split_prop)

    # 2. Shuffle the records to perform thes split. We make this first shuffle
    #    using a random_seed to allow for reproducibility of the split.
    dataset = tfrecord_data.shuffle(buffer_size=data_length, seed=random_seed)

    # 3. Decide if we use the train set of the test set
    if is_train:
        dataset = dataset.take(trainset_length)
    else:
        dataset = dataset.skip(trainset_length)

    # 4. Perform the data transormations:
    #       4.1 Apply the parsing
    #       4.2 Shuffle the final dataset
    #       4.3 Create batches of size batch_size
    dataset = (
        dataset
        .map(map_func=parser)
        .shuffle(buffer_size=trainset_length)
        .batch(batch_size=batch_size)
    )

    # 5. Create a simple iterator
    iterator = dataset.make_one_shot_iterator()

    # 6. Obtain the batch
    batch_feats, batch_labels = iterator.get_next()

    # Return the batch of (features, labels)
    return batch_feats, batch_labels


class MetaDatasetHandler(AbstractDatasetHandler):
    """The Meta-dataset Handler."""

    def __init__(self, tfrecords_rootdir, name, batch_size=256,
                 split_prop=0.33, random_seed=32):
        """Constructor."""
        self.tfrecords_rootdir = tfrecords_rootdir
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.split_prop = split_prop

        self._datasets_list = [
            "aircraft",
            "cu_birds",
            "dtd",
            "fungi",
            # "ilsvrc_2012"
            "old_quickdraw",
            "omniglot",
            "quickdraw",
            "traffic_sign",
            "vgg_flower"
        ]

        self._datasets_n_classes = [
            100,
            200,
            47,
            1394,
            345,
            1623,
            345,
            43,
            102
        ]

        # We always start with the first element in the list
        self._current_dataset = self._datasets_list[0]
        super(MetaDatasetHandler, self).__init__(name=name)

    def n_datasets(self):
        """Return the total number of datasets in the handler."""
        return len(self._datasets_list)

    def current_train_set(self):
        """Return the current train set as an input_fn."""
        return metadataset_input_fn(
            tfrecord_data=self._current_tfrecord_dataset,
            data_length=n_elements(self._current_tfrecords_files),
            batch_size=self.batch_size,
            is_train=True,
            split_prop=self.split_prop,
            random_seed=self.random_seed
        )

    def current_validation_set(self):
        """Return the current validation set as an input_fn."""
        return metadataset_input_fn(
            tfrecord_data=self._current_tfrecord_dataset,
            data_length=n_elements(self._current_tfrecords_files),
            batch_size=self.batch_size,
            is_train=False,
            split_prop=self.split_prop,
            random_seed=self.random_seed
        )

    def current_dataset_name(self):
        """Return the current dataset name."""
        return self._current_dataset

    def current_n_observations(self):
        return self._current_datalength

    def current_n_classes(self):
        return self._datasets_n_classes[
            self._datasets_list.index(self._current_dataset)
        ]

    def next_dataset(self):
        """Switch the dataset to work with."""
        next_idx = self._datasets_list.index(self._current_dataset) + 1
        if next_idx == len(self._datasets_list):
            next_idx = 0

        # Make the switch
        self._current_dataset = self._datasets_list[next_idx]

    def set_current_dataset_from_config(self):
        """Assign a dataset to work with from initial config.ini."""
        try:
            self._current_dataset = \
                CONFIG_INI[cr.SEC_METADATASET][cr.PROP_DATASETID]
            self._recompute_variables()
        except KeyError:
            raise RuntimeError(
                "No configuration has been provided for metadataset."
            )

    def set_current_dataset(self, dataset_id):
        """Assign a dataset to work with."""
        if dataset_id not in self._datasets_list:
            raise ValueError(
                "Non supported dataset '{d}'".format(d=dataset_id)
            )
        self._current_dataset = dataset_id
        self._recompute_variables()

    def _recompute_variables(self):
        # 1. List the relevant tfrecords files
        self._current_tfrecords_files = sorted(
            glob.glob(
                "{rd}/{id}/*.tfrecords".format(
                    rd=self.tfrecords_rootdir,
                    id=self.current_dataset_name()
                )
            )
        )

        # 2. Read the tfrecords and build the TFRecordDataset object
        self._current_tfrecord_dataset = tf.data.TFRecordDataset(
            self._current_tfrecords_files
        )
        self._current_datalength = n_elements(self._current_tfrecords_files)
