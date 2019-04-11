"""Test the default databases of experiments.

In the module `nasgym/database/default_db.py` we expose a simple database of
experiments bases on a CSV file. We expose simple operations that are used by
our NAS environment and we test them in this suite.
"""

import os
import unittest
import pandas as pd
from nasgym.database.default_db import DefaultExperimentsDatabase


class TestDefaultExperimentsDatabase(unittest.TestCase):
    """Test the default database of experiments."""

    def setUp(self):
        """Set up the attributes for this test case."""
        workspace_dir = "./workspace"
        self.db_path = "{workspace}/test_db.csv".format(
            workspace=workspace_dir
        )
        self.headers = ["header1", "header2", "header3"]
        self.pk_header = "header1"
        self.data = [
            ["AA", "BB", 0.],
            ["CC", "DD", 1.],
            ["FF", "GG", 2.],
            ["II", "JJ", 3.],
            ["LL", "MM", 4.],
            ["OO", "PP", 5.],
            ["RR", "SS", 6.],
            ["UU", "VV", 7.],
            ["XX", "YY", 8.]
        ]

    def test_workflow_overwrite_true(self):
        """Test that the overwrite set to true works correctly.

        The test will make sure that a new file with the new information has
        been created.
        """
        if os.path.isfile(self.db_path):
            os.remove(self.db_path)

        # Write a file
        with open(self.db_path, "w") as csv_file:
            csv_file.write("Some contect to overwrite")

        ex_database = DefaultExperimentsDatabase(
            file_name=self.db_path,
            headers=self.headers,
            pk_header=self.pk_header,
            overwrite=True
        )

        for row in self.data:
            to_add = {
                self.headers[0]: row[0],
                self.headers[1]: row[1],
                self.headers[2]: row[2]
            }
            ex_database.add(to_add)

        ex_database.save()

        pd_test = pd.read_csv(self.db_path)

        self.assertTrue(pd_test.shape[0] == len(self.data))
        self.assertTrue(pd_test.shape[1] == len(self.headers))

        for i, row in pd_test.iterrows():            
            self.assertTrue(self.data[i][0] == row[self.headers[0]])
            self.assertTrue(self.data[i][1] == row[self.headers[1]])
            self.assertTrue(self.data[i][2] == row[self.headers[2]])

    def test_workflow_overwrite_false(self):
        """Verify that the overwrite set to false works correctly."""
        if os.path.isfile(self.db_path):
            os.remove(self.db_path)

        ex_database = DefaultExperimentsDatabase(
            file_name=self.db_path,
            headers=self.headers,
            pk_header=self.pk_header,
            overwrite=False
        )

        repetitions = 2
        for _ in range(repetitions):
            for row in self.data:
                to_add = {
                    self.headers[0]: row[0],
                    self.headers[1]: row[1],
                    self.headers[2]: row[2]
                }
                ex_database.add(to_add)

            ex_database.save()

        pd_test = pd.read_csv(self.db_path)

        self.assertTrue(pd_test.shape[0] == len(self.data)*repetitions)
        self.assertTrue(pd_test.shape[1] == len(self.headers))

    def test_exists(self):
        """Test the method exists() works after adding all data."""
        if os.path.isfile(self.db_path):
            os.remove(self.db_path)

        ex_database = DefaultExperimentsDatabase(
            file_name=self.db_path,
            headers=self.headers,
            pk_header=self.pk_header,
            overwrite=True
        )

        for row in self.data:
            to_add = {
                self.headers[0]: row[0],
                self.headers[1]: row[1],
                self.headers[2]: row[2]
            }
            ex_database.add(to_add)

            ex_database.save()

        for row in self.data:
            self.assertTrue(ex_database.exists(row[0]))

        for row in self.data:
            self.assertFalse(ex_database.exists(row[0] + "ZZ"))

    def test_workflow_get_row(self):
        """Test that get_row() retrieves the right element all the time."""
        ex_database = DefaultExperimentsDatabase(
            file_name=self.db_path,
            headers=self.headers,
            pk_header=self.pk_header,
            overwrite=True
        )

        for row in self.data:
            to_add = {
                self.headers[0]: row[0],
                self.headers[1]: row[1],
                self.headers[2]: row[2]
            }
            ex_database.add(to_add)

            ex_database.save()

        for row in self.data:
            obtained = ex_database.get_row(row[0])
            self.assertTrue(len(row) == len(obtained))
            self.assertTrue(obtained[0] == row[0])
            self.assertTrue(obtained[1] == row[1])
            self.assertTrue(obtained[2] == row[2])

    def test_workflow_get_row_with_duplicates(self):
        """Test that get_row() does not break if duplicates ara present."""
        ex_database = DefaultExperimentsDatabase(
            file_name=self.db_path,
            headers=self.headers,
            pk_header=self.pk_header,
            overwrite=True
        )

        repetitions = 2
        for i in range(repetitions):                
            for row in self.data:
                to_add = {
                    self.headers[0]: row[0],
                    self.headers[1]: row[1],
                    self.headers[2]: row[2]
                }
                ex_database.add(to_add)

            ex_database.save()

        for row in self.data:
            obtained = ex_database.get_row(row[0])
            self.assertTrue(len(row) == len(obtained))
            self.assertTrue(obtained[0] == row[0])
            self.assertTrue(obtained[1] == row[1])
            self.assertTrue(obtained[2] == row[2])
