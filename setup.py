from setuptools import setup, find_packages
from os import path

setup(
    name='spam',
    version='0.0.1',
    packages=find_packages(),
    entry_points="""
    [console_scripts]
    spam=spam_classifier.cli:cli
    """,
)