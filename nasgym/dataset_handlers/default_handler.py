"""A default DatasetHandler."""

from abc import ABC, abstractmethod
import numpy as np
from nasgym.utl.miscellaneous import normalize_dataset
from nasgym.utl.miscellaneous import infer_data_shape
from nasgym.utl.miscellaneous import infer_n_classes


class AbstractDatasetHandler(ABC):
    """Abstract class for dataset handler."""

    def __init__(self, name):
        """General purpose constructor."""
        self.name = name
        super().__init__()

    @abstractmethod
    def next_dataset(self):
        """Switch to the next dataset."""

    @abstractmethod
    def n_datasets(self):
        """Return the total number of datasets in the handler."""

    @abstractmethod
    def current_train_set(self):
        """Return the current train set."""

    @abstractmethod
    def current_validation_set(self):
        """Return the current validation set."""

    @abstractmethod
    def current_dataset_name(self):
        """Return the current dataset's name."""

    @abstractmethod
    def current_n_classes(self):
        """Return the current dataset's number of classes."""

    @abstractmethod
    def current_shape(self):
        """Return the current dataset's shape."""


class DefaultDatasetHandler(AbstractDatasetHandler):
    """The Default Dataset Handler."""

    def __init__(self, train_features, train_labels, val_features, val_labels,
                 name, normalize=True, label_type=np.int32):
        """Constructor."""
        self.train_features = train_features
        self.train_labels = train_labels
        self.val_features = val_features
        self.val_labels = val_labels

        if normalize:
            self.train_features = normalize_dataset(
                dataset=self.train_features,
                baseline=255
            )
            self.val_features = normalize_dataset(
                dataset=self.val_features,
                baseline=255
            )

        self.train_labels = self.train_labels.astype(label_type)
        self.val_labels = self.val_labels.astype(label_type)

        super(DefaultDatasetHandler, self).__init__(name=name)

    def n_datasets(self):
        """Return the total number of datasets in the handler."""
        return 1

    def current_train_set(self):
        """Return the current train set."""
        return self.train_features, self.train_labels

    def current_validation_set(self):
        """Return the current validation set."""
        return self.val_features, self.val_labels

    def current_dataset_name(self):
        """Return the current dataset name."""
        return self.name

    def next_dataset(self):
        """Do nothing."""
        # pylint: disable=unnecessary-pass
        pass

    def current_n_classes(self):
        """Return the current dataset's number of classes."""
        infer_n_classes(self.train_labels)

    def current_shape(self):
        """Return the current dataset's shape."""
        return infer_data_shape(self.train_features)
