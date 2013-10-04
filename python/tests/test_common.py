from spartan import util, config
from spartan.config import flags
import imp
import multiprocessing
import os.path
import socket
import spartan
import subprocess
import sys
import time
import types
from os.path import basename, splitext
import unittest

config.add_flag('test_filter', default='')
config.add_flag('num_workers', default=4, type=int)

config.add_bool_flag('cluster', default=False)

def worker_loop(port): 
  watchdog = util.FileWatchdog()
  watchdog.start()
  spartan.start_worker('%s:9999' % socket.gethostname(), port)
  while 1:
    time.sleep(1)
  
def start_cluster_worker(i):
  t = 0
  for worker, count in config.HOSTS:
    if t + count > i: break
    t += count
  
  util.log('Starting worker %d on host %s', i, worker)
  args = ['ssh', 
          '-oForwardX11=no',
          worker,
          'cd %s && ' % os.path.abspath(os.path.curdir),
          #'xterm', '-e',
          #'gdb', '-ex', 'run', '--args',
          'python', '-m spartan.worker',
          '--master=%s:9999' % socket.gethostname(),
          '--port=%d' % (10000 + i)]
  
  time.sleep(0.1)
  p = subprocess.Popen(args, executable='ssh')
  return p

def start_cluster():
  master = spartan.start_master(9999, flags.num_workers)
  spartan.set_log_level(flags.log_level)
  time.sleep(0.1)
  for i in range(flags.num_workers):
    if flags.cluster:
      start_cluster_worker(i)
    else:
      spartan.start_worker('%s:9999' % socket.gethostname(),  10000 + i)
      
  return master

def run_cluster_tests(filename):
  util.log('Loading tests from %s', filename)
  _, argv = config.parse_known_args(sys.argv)
  
  util.log('Rest: %s', argv)
  
  mod_name, _ = splitext(basename(filename))
  module = imp.load_source(mod_name, filename)
  
  tests = [k for k in dir(module) if (
             (k.startswith('test_') or k.startswith('benchmark_')) and 
             isinstance(getattr(module, k), types.FunctionType))
          ]
  
  if flags.test_filter:
    tests = [t for t in tests if flags.test_filter in t]
    
  
  util.log('Running tests for module: %s (%s)', module, filename)
  util.log('Tests to run: %s', tests)
  master = start_cluster()
  
  for testname in tests:
    util.log('Running %s', testname)
    getattr(module, testname)(master)
  
  if flags.profile_kernels:
    spartan.api.PROF.dump_stats('kernel_prof.out')
  
def declare_test(fn):
  '''Function decorator: lift a top-level function into a TestCase'''
  class TestCase(unittest.TestCase):
    def test_fn(self):
      fn()
  return TestCase

if __name__ == '__main__':
  raise Exception, 'Should not be run directly.'