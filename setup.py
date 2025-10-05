"""Setup script to create a Python package for the project."""

from setuptools import find_packages, setup

setup(
    name="jet_nemotron_nvidia",
    version="1.0.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
)
