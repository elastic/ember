#!/usr/bin/env python

import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="UTF-8") as f:
    readme = f.read()

version = "0.1.0"
package_data = {}
setup(
    name="ember",
    version=version,
    description="Endgame Malware BEnchmark for Research",
    long_description=readme,
    packages=["ember"],
    package_data=package_data,
    author_email="proth@endgame.com")
