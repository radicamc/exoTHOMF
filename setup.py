#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='exothomf',
      version='1.0.0',
      license='MIT',
      author='Michael Radica',
      author_email='radicamc@uchicago.edu',
      packages=['exothomf'],
      include_package_data=True,
      url='https://github.com/radicamc/exoTHOMF',
      description='Tools for Stellar Spectrum and Contamination Fitting',
      package_data={'': ['README.md', 'LICENSE']},
      install_requires=['astropy', 'corner', 'dynesty', 'emcee', 'h5py', 'matplotlib',
                        'numpy==1.24.4', 'pandas', 'requests', 'scipy', 'spectres', 'tqdm'],
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
        ],
      )
