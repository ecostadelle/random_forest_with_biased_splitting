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
          'torch==1.13.1',
          'torchvision==0.14.1',
          'scikit-learn==0.24.2',
          'pandas==1.3.0',
          'fairlearn==0.9.0',
          'folktables==0.0.12',
          'frozendict==2.4.7',
          'rtdl==0.0.13',
          'xport==3.2.1',
          'tqdm==4.67.3',
          'hyperopt==0.2.7',
          'h5py==3.14.0',
          'tables==3.8.0',
          'category_encoders==2.6.4',
          'einops==0.8.2',
          'tab-transformer-pytorch==0.3.0',
          'openpyxl==3.1.5',
          'optuna==4.8.0',
          'kaggle==1.7.4.5',
          'datasets==3.6.0',
          'torchinfo==1.8.0'
      ]
      )
