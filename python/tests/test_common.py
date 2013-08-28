from spartan import util, config
from spartan.config import flags
import imp
import multiprocessing
import socket
import spartan
import sys
import time
import types

config.add_flag('test_filter', default='')
config.add_flag('num_workers', default=4, type=int)
config.add_bool_flag('multiprocess', default=False)

def worker_loop(port): 
  watchdog = util.FileWatchdog()
  watchdog.start()
  spartan.start_worker('%s:9999' % socket.gethostname(), port)
  while 1:
    time.sleep(1)
  

def start_cluster():
  spartan.set_log_level(spartan.INFO)
  master = spartan.start_master(9999, flags.num_workers)
  time.sleep(0.1)
  for i in range(flags.num_workers):
    if flags.multiprocess:
      p = multiprocessing.Process(target = lambda: worker_loop(10000 + i))
      p.daemon = True
      p.start()
    else:
      spartan.start_worker('%s:9999' % socket.gethostname(),  10000 + i)
  return master

def run_cluster_tests(filename):
  util.log('Loading tests from %s', filename)
  _, argv = config.parse_known_args(sys.argv)
  
  util.log('Rest: %s', argv)
  
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
  
  if flags.profile_kernels:
    spartan.api.PROF.dump_stats('kernel_prof.out')
  

if __name__ == '__main__':
  raise Exception, 'Should not be run directly.'