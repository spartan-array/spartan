#!/usr/bin/python

from setuptools import setup, Extension
from Cython.Build import cythonize
import os
import subprocess

cmdclass = {}
from Cython.Distutils import build_ext
cmdclass['build_ext'] = build_ext

# TODO: support sdist 
#try:
  #from Cython.Distutils import build_ext
  #cmdclass['build_ext'] = build_ext
  #suffix = '.pyx'
#except:
  #suffix = '.cpp'

## Ensure Cython .c files are up to date before uploading
#def build_cython():
  #print '#' * 10, 'Cythonizing extensions.'
  #for f in os.popen('find spartan -name "*.pyx"').read().split('\n'):
    #if not f.strip(): continue
    #print'#' * 10, 'Cythonizing %s' % f
    #assert os.system('cython --cplus "%s"' % f) == 0

#from distutils.command.sdist import sdist
#class cython_sdist(sdist):
  #'''Build Cython .c files for source distribution.'''
  #def run(self):
    #build_cython()
    #sdist.run(self)
  
#cmdclass['sdist'] = cython_sdist

# FIXME: Should integrate with setuptool
subprocess.call("make -C spartan/src", shell=True)
subprocess.call("cp spartan/src/worker spartan", shell=True)
subprocess.call("cp spartan/src/libspartan_array.so spartan", shell=True)

base = '.' #os.path.dirname(os.path.realpath(__file__))
ext_include_dirs = ['/usr/local/include', 
                    base + '/spartan/src/core',
                    base + '/spartan/src/array',
                    base + '/spartan/src/fastrpc/simple-rpc',
                    base + '/spartan/src/fastrpc/base-utils', 
                    base + '/spartan/src/fastrpc/simple-rpc/build',]
ext_link_dirs = ['/usr/lib',
                 base + '/spartan/src/core',
                 base + '/spartan/src/fastrpc/base-utils/build',
                 base + '/spartan/src/fastrpc/simple-rpc/build',]

setup(
  name='spartan',
  version='0.10',
  maintainer='Russell Power',
  maintainer_email='russell.power@gmail.com',
  url='https://github.com/spartan-array/spartan',
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
  install_requires=[
    'appdirs',
    'scipy',
    'numpy',
    'cython',
    'psutil',
    'traits',
    #'parakeet',
  ],
  package_dir={'': '.'},
  packages=['spartan', 
            'spartan.expr',
            'spartan.fastrpc',
            'spartan.array'],

  # Our extensions are written by Cython and Python C APIs
  ext_modules = [
    # Core extensions, Python APIs part.
    Extension('spartan.array._cextent_py_if',
             ['spartan/src/array/_cextent_py_if.cc'],
             language='c++',
             include_dirs = ext_include_dirs,
             library_dirs = ext_link_dirs,
             extra_compile_args=["-std=c++0x", "-lspartan_array"],
             extra_link_args=["-std=c++11", "-lspartan_array", "-lpython2.7"]),
    Extension('spartan.array._ctile_py_if',
             ['spartan/src/array/_ctile_py_if.cc'],
             language='c++',
             include_dirs = ext_include_dirs,
             library_dirs = ext_link_dirs,
             extra_compile_args=["-std=c++0x", "-lsparta_array"], 
             extra_link_args=["-std=c++11", "-lspartan_array", "-lpython2.7"]),

    # Core extensions, cython part.
    Extension('spartan.fastrpc.serialization_buffer', 
             ['spartan/fastrpc/serialization_buffer.pyx']),
    Extension('spartan.fastrpc.cloudpickle', 
             ['spartan/fastrpc/cloudpickle.pyx']),
    Extension('spartan.blob_ctx', 
             ['spartan/blob_ctx.pyx'], 
             language='c++',
             include_dirs = ext_include_dirs,
             library_dirs = ext_link_dirs,
             extra_compile_args=["-std=c++0x", "-lsimplerpc"], 
             extra_link_args=["-std=c++11", "-lsimplerpc", "-lbase", "-lpython2.7"]),
   Extension('spartan.array.sparse', 
            ['spartan/array/sparse.pyx'], 
            language='c++', 
            extra_compile_args=["-std=c++0x"], 
            extra_link_args=["-std=c++11"]),
   Extension('spartan.config', 
            ['spartan/config.pyx'], 
            language='c++',
            include_dirs = ext_include_dirs,
            library_dirs = ext_link_dirs),

   # Example extensions
   Extension('spartan.examples.netflix_core', ['spartan/examples/netflix_core.pyx']),
   Extension('spartan.examples.cf.helper', ['spartan/examples/cf/helper.pyx']),
   Extension('spartan.examples.sklearn.util.graph_shortest_path',
            ['spartan/examples/sklearn/util/graph_shortest_path.pyx']),
  ],

  cmdclass=cmdclass,
)

