# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from os import path


with open(path.join(path.abspath(path.dirname(__file__)), 'README.rst'), 'r') as readme:
    long_description = readme.read()


setup(
    name='visualization',
    description='Deep Convolutional Neural Network Visaulization',
    long_description=long_description,
    author='Ehsan Nezhadian',
    version='0.1.0',
    license='MIT',

    packages=find_packages(exclude=[]),
    # Package dependencies:
    install_requires=[
        'numpy',
        'pillow',
        'scipy',
        'tensorflow'
    ],
    # Development dependencies:
    extras_require={
        'dev': [
            'jupyter',
        ]
    }
)
