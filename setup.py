import os
import sys
import shutil
from setuptools import setup, find_packages, find_namespace_packages
from setuptools.command.install import install

setup(name='synthpy',
      version='1.0.0',
      description='Stellar synthetic spectrum wrapper',
      author='David Nidever',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/synthpy',
      scripts=['bin/spyderwebb'],
      requires=['numpy','astropy(>=4.0)','scipy','matplotlib'],
      zip_safe = False,
      include_package_data=True,
      packages=find_namespace_packages(where="python"),
      package_dir={"": "python"}      
)
