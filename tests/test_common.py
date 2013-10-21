from os.path import basename, splitext
from spartan import util, config, wrap
from spartan.cluster import start_cluster
from spartan.config import flags
import cProfile
import imp
import spartan
import sys
import time
import types
import unittest

CTX = None
def get_cluster_ctx():
  global CTX
  if CTX is None:
    config.parse_known_args(sys.argv)
    print flags.cluster
    CTX = start_cluster(flags.num_workers, flags.cluster)
    
  return CTX

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
  
def run(filename):
  config.add_flag('worker_list', type=str, default='4,8,16,32,64,80')
  config.parse_known_args(sys.argv)
  mod_name, _ = splitext(basename(filename))
  module = imp.load_source(mod_name, filename)
  util.log_info('Running benchmarks for module: %s (%s)', module, filename)
 
  if flags.profile_master:
    prof = cProfile.Profile()
    prof.enable()
  
  benchmarks = [k for k in dir(module) if (
             k.startswith('benchmark_') and 
             isinstance(getattr(module, k), types.FunctionType))
          ]
 
  if benchmarks:
    # csv header
    print 'num_workers,bench,time'
    if flags.cluster:
      workers = [int(w) for w in flags.worker_list.split(',')]
      #workers = [4, 16, 25, 36, 64, 81]
      #workers = [flags.num_workers]
    else:
      workers = [flags.num_workers]
    
    for i in workers:
      timer = BenchTimer(i)
      util.log_info('Running benchmarks on %d workers', i)
      master = start_cluster(i, local=not flags.cluster)
      run_benchmarks(module, benchmarks, master, timer)
      
      del master
  
  if flags.profile_master:  
    prof.disable()
    prof.dump_stats('master_prof.out')
  
  if flags.profile_kernels:
    join_profiles('./_kernel-profiles')


def with_ctx(fn):
  '''
  Decorator: invoke this test using a cluster instance.
  :param fn:
  '''
  def test_fn():
    ctx = get_cluster_ctx()
    fn(ctx)
      
  test_fn.__name__ = fn.__name__
  return test_fn
 
 
def join_profiles(dir):
  import glob
  import pstats
  return pstats.Stats(*glob.glob(dir + '/*')).dump_stats('./kernel-prof.out')

if __name__ == '__main__':
  raise Exception, 'Should not be run directly.'
