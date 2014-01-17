#!/usr/bin/python

from setuptools import setup, Extension
import os

cmdclass = {}

try:
  from Cython.Distutils import build_ext
  cmdclass['build_ext'] = build_ext
  suffix = '.pyx'
except:
  suffix = '.c'

# Ensure Cython .c files are up to date before uploading
def build_cython():
  print '#' * 10, 'Cythonizing extensions.'
  for f in os.popen('find spartan -name "*.pyx"').read().split('\n'):
    if not f.strip(): continue
    print '#' * 10, 'Cythonizing %s' % f
    assert os.system('cython "%s"' % f) == 0

build_cython()
from distutils.command.sdist import sdist
class cython_sdist(sdist):
  '''Build Cython .c files for source distribution.'''
  def run(self):
    build_cython()
    sdist.run(self)
  
cmdclass['sdist'] = cython_sdist

setup(
  name='spartan',
  version='0.06',
  maintainer='Russell Power',
  maintainer_email='russell.power@gmail.com',
  url='http://github.com/rjpower/spartan',
  install_requires=[
    'appdirs',
    'yappi',
    'numpy',
    'cython',
	'pyzmq',
    #'sphinx_bootstrap_theme',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Environment :: Other Environment',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: BSD License',
    'Operating System :: POSIX',
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
  ext_modules = [
    Extension('spartan.core', ['spartan/core' + suffix]),
    Extension('spartan.examples.netflix_core', ['spartan/examples/netflix_core' + suffix]),
    Extension('spartan.sparse_update', ['spartan/sparse_update' + suffix]),
    Extension('spartan.array.slicing', ['spartan/array/slicing' + suffix]),
  ],
  cmdclass=cmdclass,
)
