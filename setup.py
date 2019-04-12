"""Setup the nasgym pkg."""

from setuptools import setup, find_packages

# print(find_packages(exclude=["*.test", "*.test.*", "test.*", "test"]))

setup(
    name='nasgym',
    version='0.0.1',
    install_requires=['gym', 'tensorflow', 'pyyaml', 'pandas'],
    packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test"]),
    # metadata to display on PyPI
    author="Jorge Gomez Robles",
    author_email="j.gomezrb.dev@gmail.com",
    description="An OpenAI Gym environment for Neural Architecture Search",
    license="MIT",
    keywords="NAS reinforcement-learning openai-gym",
    # url="http://example.com/HelloWorld/",   # project home page, if any
)
