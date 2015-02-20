#!/usr/bin/python

from setuptools import setup, Extension
from distutils.command.sdist import sdist
import os

cmdclass = {}

try:
  from Cython.Distutils import build_ext
  cmdclass['build_ext'] = build_ext
  suffix = '.pyx'
except:
  suffix = '.cpp'


# Ensure Cython .c files are up to date before uploading
def build_cython():
  print '#' * 10, 'Cythonizing extensions.'
  for f in os.popen('find spartan -name "*.pyx"').read().split('\n'):
    if not f.strip(): continue
    print'#' * 10, 'Cythonizing %s' % f
    assert os.system('cython --cplus "%s"' % f) == 0


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
    'numpy',
    'cython',
    'pyzmq',
    'psutil',
    'traits',
    # 'yappi',
    # 'parakeet',
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
    Extension('spartan.examples.cf.helper', ['spartan/examples/cf/helper' + suffix]),
    Extension('spartan.rpc.serialization_buffer',
              ['spartan/rpc/serialization_buffer' + suffix]),
    Extension('spartan.cloudpickle', ['spartan/cloudpickle' + suffix]),
    Extension('spartan.rpc.serialization',
              ['spartan/rpc/serialization' + suffix],
              language='c++',
              extra_compile_args=["-std=c++0x"],
              extra_link_args=["-std=c++11"]),
    Extension('spartan.rpc.rlock',
              ['spartan/rpc/rlock' + suffix], language="c++"),
    Extension('spartan.examples.sklearn.util.graph_shortest_path',
              ['spartan/examples/sklearn/util/graph_shortest_path' + suffix]),
    Extension('spartan.array.sparse',
              ['spartan/array/sparse' + suffix],
              language='c++',
              extra_compile_args=["-std=c++0x"],
              extra_link_args=["-std=c++11"]),
    Extension('spartan.array.extent', ['spartan/array/extent' + suffix]),
    Extension('spartan.array.tile', ['spartan/array/tile' + suffix]),
    Extension('spartan.expr.operator.tiling',
              sources=['spartan/expr/operator/tiling.cc'],
              language='c++',
              extra_compile_args=["-std=c++0x"],
              extra_link_args=["-std=c++11", "-fPIC"]),
  ],
  cmdclass=cmdclass,
)
