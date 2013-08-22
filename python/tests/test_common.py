import spartan
from spartan import util
import imp
import socket
import types

NUM_WORKERS = 1

def start_cluster():
  spartan.set_log_level(spartan.INFO)
  master = spartan.start_master(9999, NUM_WORKERS)
  for i in range(NUM_WORKERS):
    spartan.start_worker('%s:9999' % socket.gethostname(),  10000 + i)
  return master

def run_cluster_tests(filename):
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--test_filter', default='')
  
  flags, argv = parser.parse_known_args()
  
  module = imp.load_source('test_module', filename)
  
  tests = [k for k in dir(module) if (
             k.startswith('test_') and 
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
  
  


if __name__ == '__main__':
  raise Exception, 'Should not be run directly.'