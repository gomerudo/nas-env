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


class DefaultDatasetHandler(AbstractDatasetHandler):
    """The Default Dataset Handler."""

    def __init__(self, train_X, train_y, val_X, val_y, name):
        """Constructor."""
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y

        super(DefaultDatasetHandler, self).__init__(name=name)

    def n_datasets(self):
        """Return the total number of datasets in the handler."""
        return 1

    def current_train_set(self):
        """Return the current train set."""
        return self.train_X, self.train_y

    def current_validation_set(self):
        """Return the current validation set."""
        return self.val_X, self.val_y

    def next_dataset(self):
        """Do nothing."""
        pass
