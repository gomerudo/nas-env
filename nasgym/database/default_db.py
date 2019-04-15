"""Default Database class."""

import os
import pandas as pd


class DefaultExperimentsDatabase:
    """Default database with a CSV."""

    def __init__(self, file_name, headers, pk_header, overwrite=False):
        """Constructor."""
        self.file_name = file_name
        self.headers = headers
        self.pk_header = pk_header

        if self.pk_header not in self.headers:
            raise ValueError("Invalid pk_header: pk_header is not in headers.")

        # If overwrite is passed as True and the file already exists: delete it
        if overwrite and os.path.isfile(self.file_name):
            os.remove(self.file_name)

        if os.path.isfile(self.file_name):
            self._internal_df = pd.read_csv(self.file_name)
        else:
            self._internal_df = pd.DataFrame(columns=headers)

    def exists(self, pk_value):
        """Check if a row with the given PK already exists."""
        pk_column = self._internal_df[self.pk_header]
        exists = not pk_column[pk_column.isin([pk_value])].empty
        return exists

    def get_row(self, pk_value):
        """Check if a row with the given PK already exists."""
        pk_column = self._internal_df[self.pk_header]
        res = self._internal_df[pk_column.isin([pk_value])]
        row_dict = {}
        for _, row in res.iterrows():
            row_dict = row
            break
        return row_dict

    def add(self, row):
        """Add a row to the Data base."""
        if not isinstance(row, dict):
            raise TypeError("Value to store must be of type 'dict'")

        # Append the dictionary, ignoring index
        self._internal_df = self._internal_df.append(row, ignore_index=True)

    def save(self):
        """Save the current Data Frame as a file."""
        self._internal_df.to_csv(self.file_name, index=False)
