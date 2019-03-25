"""Diverse methods for the environment management."""


def is_valid_config_file(config_file):
    """Verify whether or not the configuration file is valid."""

    return config_file.lower().endswith('.yml')

