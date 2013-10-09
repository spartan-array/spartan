from os.path import basename, splitext
from spartan import util, config
from spartan.cluster import start_cluster
from spartan.config import flags
import cProfile
import imp
import spartan
import sys
import time
import types
import unittest


config.add_flag('test_filter', default='')

class BenchTimer(object):
  def __init__(self, num_workers):
    self.times = []
    self.num_workers = num_workers
     
  def time_op(self, key, fn):
    st = time.time()
    fn()
    ed = time.time()
    print '%d,"%s",%f' % (self.num_workers, key, ed - st)
    

def run_benchmarks(module, benchmarks, master, timer):
  time.sleep(0.1)
  for benchname in benchmarks:
    getattr(module, benchname)(master, timer)
  
def run_tests(module):
  tests = [k for k in dir(module) if (
             k.startswith('test_') and 
             isinstance(getattr(module, k), types.FunctionType))
          ]
  
  if flags.test_filter:
    tests = [t for t in tests if flags.test_filter in t]
    
  if not tests:
    return
  
  master = start_cluster(flags.num_workers, not flags.cluster)
  util.log_warn('Tests to run: %s', tests)
  for testname in tests:
    util.log_warn('Running %s', testname)
    getattr(module, testname)(master)

def run(filename):
  util.log('Loading tests from %s', filename)
  _, argv = config.parse_known_args(sys.argv)
  
  util.log('Rest: %s', argv)
  
  mod_name, _ = splitext(basename(filename))
  module = imp.load_source(mod_name, filename)
  util.log('Running tests for module: %s (%s)', module, filename)
  
  run_tests(module)
 
  if flags.profile_master:
    prof = cProfile.Profile()
    prof.enable()
  
  benchmarks = [k for k in dir(module) if (
             k.startswith('benchmark_') and 
             isinstance(getattr(module, k), types.FunctionType))
          ]
 
  if benchmarks:
    # header
    print 'num_workers,bench,time'
    if flags.cluster:
      workers = [1, 2, 4, 8, 16, 32, 64, 80]
    else:
      workers = [flags.num_workers]
    
    for i in workers:
      timer = BenchTimer(i)
      util.log('Running benchmarks on %d workers', i)
      master = start_cluster(i, local=not flags.cluster)
      run_benchmarks(module, benchmarks, master, timer)
      
      del master
  
  if flags.profile_master:  
    prof.disable()
    prof.dump_stats('master_prof.out')
  
  if flags.profile_kernels:
    spartan.PROF.dump_stats('kernel_prof.out')
  
def declare_test(fn):
  '''Function decorator: lift a top-level function into a TestCase'''
  class TestCase(unittest.TestCase):
    def test_fn(self):
      fn()
  return TestCase

if __name__ == '__main__':
  raise Exception, 'Should not be run directly.'