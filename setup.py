#!/usr/bin/python

from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
  name='spartan',
  version='0.06',
  maintainer='Russell Power',
  maintainer_email='russell.power@gmail.com',
  url='http://github.com/rjpower/spartan',
  install_requires=[
    'appdirs',
    'numpy>=1.6',
    'cython',
    'sphinx_bootstrap_theme',                
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Environment :: Other Environment',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: BSD License',
    'Operating System :: POSIX',
    'Programming Language :: C++',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.6',
    'Programming Language :: Python :: 2.7',
  ],
  description='Distributed Numpy-like arrays.',
  package_dir={'': '.'},
  packages=['spartan', 
            'spartan.expr',
            'spartan.rpc',
            'spartan.array' ],
  ext_modules = cythonize(
              'spartan/core.pyx', 
              'spartan/examples/netflix_core.pyx')
)
