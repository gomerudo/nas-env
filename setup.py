"""Setup the nasgym pkg with setuptools."""

from setuptools import setup, find_packages

setup(
    # For installation
    name='nasgym',
    version='0.0.1',
    install_requires=['gym', 'tensorflow', 'pyyaml', 'pandas'],
    packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test"]),

    # Metadata to display on PyPI
    author="Jorge Gomez Robles",
    author_email="j.gomezrb.dev@gmail.com",
    description="An OpenAI Gym environment for Neural Architecture Search",
    license="MIT",
    keywords="NAS reinforcement-learning openai-gym",
    url="https://gomerudo.github.io/nas-env/",
)
