#!/usr/bin/env python

# This file was modified from the original version.
# Original project licensed under the MIT License.
# Changes made on 2026-04-17.


from distutils.core import setup
from setuptools import find_packages

setup(name='tableshift',
      version='0.1',
      url='https://tableshift.org',
      description='A tabular data benchmarking toolkit.',
      long_description='A benchmarking toolkit for tabular data under distirbution shift. '
                       'For more details, see the paper '
                       '"Benchmarking Distribution Shift in Tabular Data with TableShift", '
                       'Gardner, Popovic, and Schmidt, 2023.',
      author='Josh Gardner',
      author_email='jpgard@cs.washington.edu',
      packages=find_packages(),
      include_package_data=True,
      data_files=[('tableshift/datasets',
                   ['tableshift/datasets/nhanes_data_sources.json',
                    'tableshift/datasets/icd9-codes.json'])],
      install_requires=[
          'numpy==1.19.5',
          'ray==2.2',
          'torch',
          'torchvision',
          'scikit-learn==0.24.2',
          'pandas==1.3.0',
          'fairlearn',
          'folktables',
          'frozendict',
          'rtdl',
          'xport==3.2.1',
          'tqdm',
          'hyperopt',
          'h5py',
          'tables==3.8.0',
          'category_encoders',
          'einops',
          'tab-transformer-pytorch',
          'openpyxl',
          'optuna',
          'kaggle',
          'datasets==3.6.0',
          'torchinfo'
      ]
      )
