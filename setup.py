import os
from setuptools import find_packages, setup

setup(
    name="sr_metrics",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "lpips",
        "numpy",
        "pillow",
        "tqdm"
    ],
)
