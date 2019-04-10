"""Test the creation of the network."""

import unittest
from nasgym.database.default_db import DefaultExperimentsDatabase


class TestDefaultExperimentsDatabase(unittest.TestCase):
    """Test the parsing of Neural Structure Code (NSC) to Network with TF."""

    def test_workflow_override(self):
        ex_database = DefaultExperimentsDatabase(
            file_name="workspace/db.csv",
            headers=["a", "b", "c"],
            pk_header="a",
            overwrite=True
        )

        for a, b, c in zip(range(10), range(10), range(10)):
            to_add = {'a': a, 'b': b, 'c': c}
            ex_database.add(to_add)
            ex_database.save()

    def test_workflow_notoverride(self):
        ex_database = DefaultExperimentsDatabase(
            file_name="workspace/db.csv",
            headers=["a", "b", "c"],
            pk_header="a",
            overwrite=False
        )

        for a, b, c in zip(range(10), range(10), range(10)):
            to_add = {'a': a, 'b': b, 'c': c}
            ex_database.add(to_add)

        ex_database.save()

    def test_workflow_exists(self):
        ex_database = DefaultExperimentsDatabase(
            file_name="workspace/db.csv",
            headers=["a", "b", "c"],
            pk_header="a",
            overwrite=True
        )

        for a, b, c in zip(range(10), range(10), range(10)):
            to_add = {'a': str(a), 'b': b, 'c': c}
            ex_database.add(to_add)

        ex_database.save()
        if ex_database.exists("0"):
            print(ex_database.get_row("0"))
        # print(ex_database.exists(0))
