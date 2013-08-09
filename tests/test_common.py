from pytable import util
import imp
import os
import subprocess
import sys
import types
import shlex


def run_cluster_tests(filename):
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--with_gdb', default=False, action='store_true')
  parser.add_argument('--with_valgrind', default=False, action='store_true')
  parser.add_argument('--use_cluster', default=False, action='store_true')
  parser.add_argument('--test_filter', default='')
  parser.add_argument('--internal_run_test', default='')
  
  flags, argv = parser.parse_known_args()
  
  module = imp.load_source('test_module', filename)
  
  if flags.internal_run_test:
    import pytable
    argv = ['python'] + argv
    master = pytable.init(argv)
    print >> sys.stderr, 'RUNNING: ', flags.internal_run_test
    fn = getattr(module, flags.internal_run_test)
    fn(master)
    return
  
  # setup environment, and call MPI for each test
  env = dict(os.environ)
  env['PYTHONPATH'] += ':build/.libs/:build/pytable'
  
  tests = [k for k in dir(module) if (
             k.startswith('test_') and 
             isinstance(getattr(module, k), types.FunctionType))
          ]
  
  if flags.test_filter:
    tests = [t for t in tests if flags.test_filter in t]
    
  
  util.log('Running tests for module: %s (%s)', module, filename)
  util.log('Tests to run: %s', tests)

  args = [ 'mpirun', 
          '-n', '64' if flags.use_cluster else '4', 
          '--bynode',
           '--mca', 'btl', 'tcp,self',
           '--mca', 'btl_tcp_if_include', '216.165.108.0/24',
           '-x', 'PYTHONPATH',
        ]


  mpi_debug = flags.with_gdb and flags.use_cluster
  if mpi_debug:
    args += ['-d', '--mca', 'pls_rsh_agent', 'ssh -X -n', ]
  
  if flags.use_cluster:
    args += [ '-hostfile', 'conf/mpi-cluster' ]
  else: 
    args += [ '-hostfile', 'conf/mpi-local', ]
   
  if flags.with_gdb or flags.with_valgrind:
    args += [ 'xterm', '-e', ]

  if flags.with_gdb:
    args = args + ['gdb', '-ex', 'run', '--args' ]
  elif flags.with_valgrind:
    args = args + ['valgrind', sys.executable, filename]
    
  
  for t in tests:
    proc_args = args + [sys.executable, filename, '--internal_run_test=%s' % t] 
    proc_args += shlex.split(' '.join(argv))
    print proc_args

    p = subprocess.Popen(env = env, args=proc_args)
      
    if p.wait() != 0:
      util.log('Test %s FAILED', t)


if __name__ == '__main__':
  raise Exception, 'Should not be run directly.'