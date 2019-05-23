import unittest
from nasgym.utl.configreader import read_configfile


class TestConfigReader(unittest.TestCase):
    """Test the default database of experiments."""

    def test_read_configfile(self):
        config_dict = read_configfile()
        print(config_dict)
