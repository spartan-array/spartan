import os
from os.path import basename, splitext
import signal
from spartan import util, config
from spartan.cluster import start_cluster
from spartan.config import FLAGS, StrFlag
import cProfile
import imp
import spartan
import sys
import time
import types
import unittest
from spartan.array import distarray

CTX = None
def get_cluster_ctx():
  global CTX
  if CTX is None:
    util.log_info('Starting cluster...')
    config.initialize(sys.argv)
    CTX = start_cluster(FLAGS.num_workers, FLAGS.cluster)
  return CTX

def sig_handler(sig, frame):
  import threading
  import sys
  import traceback

  for thread_id, stack in sys._current_frames().items():
    print '-' * 100
    traceback.print_stack(stack)

class BenchTimer(object):
  def __init__(self, num_workers):
    self.times = []
    self.num_workers = num_workers
     
  def time_op(self, key, fn):
    st = time.time()
    result = fn()
    ed = time.time()
    print '%d,"%s",%f' % (self.num_workers, key, ed - st)
    return result
    

def run_benchmarks(module, benchmarks, master, timer):
  time.sleep(0.1)
  for benchname in benchmarks:
    getattr(module, benchname)(master, timer)
  
def run(filename):
  signal.signal(signal.SIGQUIT, sig_handler)
  os.system('rm ./_kernel-profiles/*')

  FLAGS.add(StrFlag('worker_list', default='4,8,16,32,64,80'))

  config.initialize(sys.argv)
  mod_name, _ = splitext(basename(filename))
  module = imp.load_source(mod_name, filename)
  util.log_info('Running benchmarks for module: %s (%s)', module, filename)
 
  if FLAGS.profile_master:
    prof = cProfile.Profile()
    prof.enable()
  
  benchmarks = [k for k in dir(module) if (
             k.startswith('benchmark_') and 
             isinstance(getattr(module, k), types.FunctionType))
          ]
 
  if benchmarks:
    # csv header
    print 'num_workers,bench,time'
    workers = [int(w) for w in FLAGS.worker_list.split(',')]
    
    for i in workers:
      timer = BenchTimer(i)
      util.log_info('Running benchmarks on %d workers', i)
      master = start_cluster(i, FLAGS.cluster)
      run_benchmarks(module, benchmarks, master, timer)
      master.shutdown()

  if FLAGS.profile_master:
    prof.disable()
    prof.dump_stats('master_prof.out')
  
  if FLAGS.profile_kernels:
    join_profiles('./_kernel-profiles')


class ClusterTest(unittest.TestCase):
  '''
  Helper class for running cluster tests.
  
  Ensures a cluster instance is available before running any test
  cases, and resets the TILE_SIZE for distarray between test runs.
  '''
  TILE_SIZE = None
  flags = None
  
  @classmethod
  def setUpClass(cls):
    cls.ctx = get_cluster_ctx()
    cls.old_tilesize = distarray.TILE_SIZE
    if cls.TILE_SIZE is not None:
      distarray.TILE_SIZE = cls.TILE_SIZE

    cls.old_flags = {}
    if cls.flags is not None:
      for k, v in cls.flags.iteritems():
        cls.old_flags[k] = getattr(config.FLAGS, k)
        setattr(config.FLAGS, k, v)

  @classmethod
  def tearDownClass(cls):
    distarray.TILE_SIZE = cls.old_tilesize
    for k, v in cls.old_flags.iteritems():
      setattr(config.FLAGS, k, v)


def with_ctx(fn):
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
