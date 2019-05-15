"""A default DatasetHandler."""

from abc import ABC, abstractmethod


class AbstractDatasetHandler(ABC):
    """Abstract class for dataset handler."""

    def __init__(self, name):
        """General purpose constructor."""
        self.name = name
        super().__init__()

    @abstractmethod
    def next_dataset(self):
        """Add the training graph."""

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
        """Return the current dataset name."""


class DefaultDatasetHandler(AbstractDatasetHandler):
    """The Default Dataset Handler."""

    def __init__(self, train_features, train_labels, val_features, val_labels,
                 name):
        """Constructor."""
        self.train_features = train_features
        self.train_labels = train_labels
        self.val_features = val_features
        self.val_labels = val_labels
        print("Shape of train_features is:", self.train_features.shape)
        print("Shape of train_labels is:", self.train_labels.shape)
        print("Shape of val_features is:", self.val_features.shape)
        print("Shape of val_labels is:", self.val_labels.shape)

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
