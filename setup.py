import os, sys
import subprocess
#from distutils.core import setup, Extension, Command
from setuptools import setup, Extension, Command
from Cython.Distutils import build_ext

from distutils.sysconfig import get_python_lib
from distutils.command.install import INSTALL_SCHEMES

import shutil

class clean(Command):
  description = 'Remove build and trash files'
  user_options = [("all", "a", "the smae")]

  def initialize_options(self):
    self.all = None

  def finalize_options(self):
    pass

  def run(self):
    subprocess.call("rm -rf spartan/*.so spartan/*.c spartan/*.cpp spartan/worker spartan/lib", shell=True)
    subprocess.call("rm -rf spartan/array/*.so spartan/array/*.c spartan/array/*.cpp", shell=True)
    subprocess.call("rm -rf spartan/rpc/*.so spartan/rpc/*.c spartan/rpc/*.cpp spartan/rpc/simplerpc", shell=True)
    subprocess.call("make -C spartan/src clean", shell=True)
    subprocess.call("rm -rf build", shell=True)

#We need to build up src/ before setup invokes
#Assume we are already under /home/..../spartan
def pre_install():
  subprocess.call("make -C spartan/src", shell = True)
  subprocess.call("mkdir -p spartan/rpc/simplerpc", shell = True)
  path = os.path.join(os.getcwd(), 'spartan/src/rpc/simple-rpc/pylib/simplerpc/')
  new_path = os.path.join(os.getcwd(), 'spartan/rpc/simplerpc')

  try:
    os.mkdir('spartan/rpc/simplerpc')
  except OSError:
    pass

  for f in os.listdir(path):
    if f.endswith(".py"):
      rfp = open(os.path.join(path, f))
      wfp = open(os.path.join(new_path, f), 'w')
      for line in rfp:
        line = line.replace('simplerpc.', '.')
        line = line.replace('simplerpc ', '. ')
        wfp.write(line)

      rfp.close()
      wfp.close()

  path = os.path.join(os.getcwd(), 'spartan/src/rpc/service.py')
  new_path = os.path.join(os.getcwd(), 'spartan/rpc/simplerpc/service.py')

  rfp = open(path)
  wfp = open(new_path, 'w')

  for line in rfp:
    line = line.replace('simplerpc.', '.')
    line = line.replace('simplerpc ', '. ')
    wfp.write(line)

  rfp.close()
  wfp.close()

  #Make marshal.py into .pyx for Cython use
#  subprocess.call("mv -i spartan/rpc/simplerpc/marshal.py \
#                          spartan/rpc/simplerpc/marshal.pyx", shell = True)

def fetch_from_src(dic):
  '''
  Append executables from /src/* into metadata
  '''
  path = os.path.join(os.getcwd(), 'spartan/src')

  #Make directories
  try:
    os.mkdir('spartan/lib')
  except OSError:
    pass

  #From /src/pkg to spartan
  src_pkg = []
  for f in os.listdir(os.path.join(path, 'obj/pkg')):
    src_pkg.append('spartan/src/obj/pkg/' + f)
    shutil.copyfile(os.path.join(path, 'obj/pkg/')+f, 'spartan/'+f)
    shutil.copymode(os.path.join(path, 'obj/pkg/')+f, 'spartan/'+f)

  #From /src/lib to spartan/lib
  src_lib = []
  for f in os.listdir(os.path.join(path, 'obj/lib')):
    src_lib.append('spartan/src/obj/lib/' + f)
    shutil.copyfile(os.path.join(path, 'obj/lib/')+f, 'spartan/lib/'+f)
    shutil.copymode(os.path.join(path, 'obj/lib/')+f, 'spartan/lib/'+f)

  #From /src/rpc to spartan/rpc/simplerpc
  src_rpc = []
  for f in os.listdir(os.path.join(path, 'obj/rpc')):
    src_rpc.append('spartan/src/obj/rpc/' + f)
    shutil.copyfile(os.path.join(path, 'obj/rpc/')+f, 'spartan/rpc/simplerpc/'+f)
    shutil.copymode(os.path.join(path, 'obj/rpc/')+f, 'spartan/rpc/simplerpc/'+f)

  #Copy all pylib/simplerpc/*.py into spartan/rpc/simplerpc
  path = os.getcwd()
  for f in os.listdir(os.path.join(path, 'spartan/rpc/simplerpc')):
#    if f.endswith('.py') and not 'marshal' in f:
    if f.endswith('.py'):
      src_rpc.append(os.path.join(os.path.join(path, 'spartan/rpc/simplerpc'), f))

  dic['data_files'] = [
                        ('spartan', src_pkg),
                        ('spartan/lib', src_lib),
                        ('spartan/rpc/simplerpc', src_rpc), 
                        ('spartan/rpc', ['spartan/src/rpc/service.py'])
                      ]

def setup_package():
  src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
  old_path = os.getcwd()

  #Calling Makefile in spartan/src
  if not 'clean' in sys.argv:
    pre_install()

  #Set Spartan source path first
  os.chdir(src_path)
  sys.path.insert(0, src_path)

  #Set packages
  pkgs = [
    'spartan',
    'spartan.expr',
    'spartan.array',
    'spartan.rpc',
  ]

  #See this link for explanation:
  #https://groups.google.com/forum/#!topic/comp.lang.python/Nex7L-026uw
  for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']

  #Set rpath
  runtime_link = {
                    'spartan' : ['$ORIGIN/lib'],
                    'spartan/lib' : ['$ORIGIN/../lib'],
                    'spartan/expr' : ['$ORIGIN/../lib'],
                    'spartan/array' : ['$ORIGIN/../lib'],
                    'spartan/rpc' : ['$ORIGIN/../lib'],
                  }

  pkgs_dir = {p : p.replace('.', '/') for p in pkgs}

  ext_include_dirs = ['/usr/local/include',
                      src_path + '/spartan/src',
                      src_path + '/spartan/src/rpc/simple-rpc',
                      src_path + '/spartan/src/rpc/simple-rpc/build', ]
  ext_link_dirs = ['/usr/lib',
                  src_path + '/spartan/src/',
                  src_path + '/spartan/src/obj/pkg',
                  src_path + '/spartan/src/obj/lib',
                  src_path + '/spartan/src/obj/rpc',
                  src_path + '/spartan/src/rpc/simple-rpc/build/base',
                  src_path + '/spartan/src/rpc/simple-rpc/build', ]

  metadata = dict(
    name = 'spartan',
    version = '0.10',
    maintainer = 'Russell Power',
    maintainer_email = 'russell.power@gmail.com',
    url = 'https://github.com/spartan-array/spartan',
    classifiers = [
      'Development Status :: 4 - Beta',
      'Environment :: Other Environment',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Operating System :: POSIX',
      'Programming Language :: Python',
      'Programming Language :: Python :: 2.6',
      'Programming Language :: Python :: 2.7',
    ],
    description = 'Distributed Numpy-like arrays.',
    install_requires = [
      'appdirs',
      'scipy',
      'numpy',
      'cython',
      'psutil',
      'traits',
    ],
    packages = pkgs,
    package_dir = pkgs_dir,

    # Our extensions are written by Cython and Python C APIs
    ext_modules=[
      # Spartan extensions, Python APIs part.
      Extension('spartan.array._cextent_py_if',
                ['spartan/src/array/_cextent_py_if.cc'],
                language='c++',
                include_dirs=ext_include_dirs,
                library_dirs=ext_link_dirs,
                extra_compile_args=["-std=c++0x", "-lspartan_array"],
                extra_link_args=["-std=c++11", "-lspartan_array", "-lpython2.7"],
                runtime_library_dirs=runtime_link['spartan/array'],
                depends=["spartan/lib/libspartan_array.so"]),
      Extension('spartan.array._ctile_py_if',
                ['spartan/src/array/_ctile_py_if.cc'],
                language='c++',
                include_dirs=ext_include_dirs,
                library_dirs=ext_link_dirs,
                extra_compile_args=["-std=c++0x", "-lsparta_array"],
                extra_link_args=["-std=c++11", "-lspartan_array", "-lpython2.7"],
                runtime_library_dirs=runtime_link['spartan/array'],
                depends=["spartan/lib/libspartan_array.so"]),
      Extension('spartan._cblob_ctx_py_if',
                ['spartan/src/core/_cblob_ctx_py_if.cc'],
                language='c++',
                include_dirs=ext_include_dirs,
                library_dirs=ext_link_dirs,
                extra_compile_args=["-std=c++0x", "-lsparta_array", "-lsimplerpc", "-lcore"],
                extra_link_args=["-std=c++11", "-lspartan_array", "-lsimplerpc",
                                "-lbase", "-lcore", "-lpython2.7"],
                runtime_library_dirs=runtime_link['spartan'],
                depends=[
                          "spartan/lib/libspartan_array.so",
                          "spartan/lib/libsimplerpc.so",
                          "spartan/lib/libcore.so",
                          "spartan/lib/libbase.so"
                        ]),
      Extension('spartan.rpc._rpc_array',
                ['spartan/src/rpc/_rpc_array.cc'],
                language='c++',
                include_dirs=ext_include_dirs,
                library_dirs=ext_link_dirs,
                extra_compile_args=["-std=c++0x", "-lsparta_array", "-lsimplerpc"],
                extra_link_args=["-std=c++11", "-lspartan_array", "-lsimplerpc",
                                "-lbase", "-lpython2.7"],
                runtime_library_dirs=runtime_link['spartan/rpc'],
                depends=[
                          "spartan/lib/libspartan_array.so",
                          "spartan/lib/libsimplerpc.so",
                          "spartan/lib/libbase.so"
                        ]),
      Extension('spartan.expr.tiling',
                sources=['spartan/expr/tiling.cc'],
                language='c++',
                extra_compile_args=["-std=c++0x"],
                extra_link_args=["-std=c++11", "-fPIC"],
                runtime_library_dirs=runtime_link['spartan/expr']),

      # Spartan extensions, cython part.
      Extension('spartan.rpc.serialization_buffer',
                ['spartan/rpc/serialization_buffer.pyx'],
                extra_compile_args=["-pipe"]),
      Extension('spartan.rpc.cloudpickle',
                ['spartan/rpc/cloudpickle.pyx'],
                extra_compile_args=["-pipe"]),
#      Extension('spartan.rpc.simplerpc.marshal',
#                ['spartan/rpc/simplerpc/marshal.pyx'],
#                extra_compile_args=["-pipe"],
#                runtime_library_dirs=runtime_link),
      Extension('spartan.array.sparse',
                ['spartan/array/sparse.pyx'],
                language='c++',
                extra_compile_args=["-std=c++0x", "-pipe"],
                extra_link_args=["-std=c++11"]),
      Extension('spartan.config',
                ['spartan/config.pyx'],
                language='c++',
                include_dirs=ext_include_dirs,
                library_dirs=ext_link_dirs,
                extra_compile_args=["-std=c++0x", "-lcore", "-pipe"],
                extra_link_args=["-std=c++11", "-lcore"],
                runtime_library_dirs=runtime_link['spartan'],
                depends=["spartan/lib/libcore.so"]),

      # Example extensions
      Extension('spartan.examples.netflix_core', ['spartan/examples/netflix_core.pyx']),
      Extension('spartan.examples.cf.helper', ['spartan/examples/cf/helper.pyx']),
      Extension('spartan.examples.sklearn.util.graph_shortest_path',
                ['spartan/examples/sklearn/util/graph_shortest_path.pyx']),
    ],

    cmdclass = {
      'build_ext' : build_ext,
      'clean' : clean
    },
  )

  if not 'clean' in sys.argv:
    fetch_from_src(metadata)

  try:
    setup(**metadata)
  finally:
    del sys.path[0]
    os.chdir(old_path)
  return

if __name__ == '__main__':
  setup_package()
