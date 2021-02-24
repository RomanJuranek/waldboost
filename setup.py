#!/usr/bin/env python3

import sys
from setuptools import setup


def version():
    with open("waldboost/VERSION","r") as f:
        return f.read().strip()

setup(
    name = "waldboost",
    version = version(),
    description = "Object detection with WaldBoost for Python",
    author = "Roman Juranek",
    author_email = "ijuranek@fit.vutbr.cz",
    url = "https://github.com/RomanJuranek/waldboost",
    keywords = "object detection, waldboost",
    packages = ["waldboost"],
    install_requires = ["numpy", "scikit-learn", "scipy", "protobuf", "numba", "bbx>=0.0.7"],
    scripts = None,
    python_requires = ">=3.6",
    include_package_data = True,
    )
