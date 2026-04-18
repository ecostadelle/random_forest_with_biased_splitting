import numpy
import sys
from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages
import os

if sys.platform.startswith("win"):
    openmp_arg = '/openmp'
else:
    openmp_arg = '-fopenmp'

long_description = """
Intervention in Prediction Measure (IPM)
"""

extensions = [
    Extension(
        '*',
        ['ipm/*.pyx'],
        define_macros=[(
            'NPY_NO_DEPRECATED_API', 
            'NPY_1_7_API_VERSION'
        )],
        extra_compile_args=[openmp_arg, '-std=c++17'],
        extra_link_args=[openmp_arg],
        language='c++'
    ),
]

setup(
    name='ipm',
    version='0.1',
    author='Ewerton Costadelle',
    author_email='ecostadelle@id.uff.br',
    description='Intervention in Prediction Measure (IPM) as proposed by Epifânio (2017)',
    long_description=long_description,
    long_description_content_type='text/markdown', 
    ext_modules=cythonize(
        extensions,
        gdb_debug=True,
        language_level='3',
        compiler_directives={
                'boundscheck': False,
                'wraparound': False
        }
    ),
    include_dirs=[
        numpy.get_include(),
    ],
    zip_safe=False,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'pandas',
        'scipy',
        'pyarrow',
        'tabulate',
        'Jinja2'
    ],
    include_package_data=True,
    package_data={'ipm': ['*.pxd', '*.pyx', '*.py'] },
)