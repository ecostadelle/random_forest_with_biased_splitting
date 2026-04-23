# Copyright (C) 2022  Marcelo R. H. Maia <mmaia@ic.uff.br, marcelo.h.maia@ibge.gov.br>
# License: GPLv3 (https://www.gnu.org/licenses/gpl-3.0.html)

# This file was modified from the original version distributed under GPLv3.
# Changes made on 2026-04-17.


from setuptools import setup, Extension, find_packages

import numpy
from Cython.Build import cythonize

extensions = [Extension('*', ['maia/*.pyx'], libraries=['gsl', 'gslcblas'],
                        library_dirs=['../external_lib/gsl_x64-windows/lib'])]

setup(
    ext_modules=cythonize(extensions, language_level='2'),
    include_dirs=[numpy.get_include(), '../external_lib/gsl_x64-windows/include'],
    zip_safe=False,
    name='maia',
    version='0.1',
    author='Marcelo R. H. Maia',
    author_email='mmaia@ic.uff.br',
    description='Original Biased Splitting Random Forest implementation, by Maia (2023)',
    packages=find_packages(),
    install_requires=[
        'numpy==1.19.5',
        'scikit-learn==0.24.2'
    ],
    include_package_data=True,
    package_data={'maia': ['*.pxd', '*.pyx', '*.py'] },
)
