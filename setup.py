#!/usr/bin/env python3

import sys
from setuptools import setup


def version():
    with open("waldboost/VERSION","r") as f:
        return f.read().strip()

setup(
    name = "waldboost",
    version = version(),
    description = "Pure python/numpy object detection",
    author = "Roman Juranek",
    author_email = "ijuranek@fit.vutbr.cz",
    url = "",
    keywords = "object detection, waldboost",
    packages = ["waldboost"],
    install_requires = ["numpy", "scikit-image", "scikit-learn", "scipy", "protobuf", "numba", "opencv-python"],
    scripts = None,
    python_requires = ">=3.6",
    include_package_data = True,
    )
