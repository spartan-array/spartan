import atexit
import os
from os.path import basename, splitext
import signal
from spartan import util, config
from spartan.cluster import start_cluster
from spartan.config import FLAGS, StrFlag, BoolFlag
import cProfile
import imp
import spartan
import sys
import time
import types
import unittest

FLAGS.add(StrFlag('worker_list', default='4,8,16,32,64,80'))
FLAGS.add(BoolFlag('test_optimizations', default=False))

def millis(t1, t2):
  dt = t2 - t1
  ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
  return ms

def sig_handler(sig, frame):
  import threading
  import sys
  import traceback

  for thread_id, stack in sys._current_frames().items():
    print '-' * 100
    traceback.print_stack(stack)

class BenchTimer(object):
  def __init__(self, num_workers):
    self.num_workers = num_workers
    self.prefix = ''

  def time_op(self, key, fn):
    st = time.time()
    result = fn()
    ed = time.time()
    self.log('%d,"%s",%f', self.num_workers, key, ed - st)
    return result

  def benchmark_op(self, op, min_time=1.0):
    '''Run ``op`` in a loop until ``min_time`` has passed.'''
    iters = 0
    from time import time
    st = time()

    while 1:
        iters += 1
        op()
        ed = time()
        if ed - st > min_time:
            break

    self.log('Ran %d ops, %.3f seconds, %f s/op, %f ops/s' % (iters, ed - st, (ed - st) / iters, iters / (ed - st)))


  def log(self, fmt, *args):
    msg = fmt % args if len(args) > 0 else fmt
    print self.prefix, msg
    

def run_benchmarks(module, benchmarks, master, timer):
  for benchname in benchmarks:
    getattr(module, benchname)(master, timer)

def benchmark_op(op, min_time=1.0):
  '''Run ``op`` in a loop until ``min_time`` has passed.'''
  iters = 0
  from time import time
  st = time()

  while 1:
      iters += 1
      op()
      ed = time()
      if ed - st > min_time:
          break

  print 'Ran %d ops, %.3f seconds, %f s/op, %f ops/s' % (iters, ed - st, (ed - st) / iters, iters / (ed - st))

  
def run(filename):
  signal.signal(signal.SIGQUIT, sig_handler)
  os.system('rm ./_worker_profiles/*')

  mod_name, _ = splitext(basename(filename))
  module = imp.load_source(mod_name, filename)
  util.log_info('Running benchmarks for module: %s (%s)', module, filename)
  benchmarks = [k for k in dir(module) if (
             k.startswith('benchmark_') and 
             isinstance(getattr(module, k), types.FunctionType))
          ]

  spartan.config.parse(sys.argv)
  if benchmarks:
    # csv header
    print 'num_workers,bench,time'
    workers = [int(w) for w in FLAGS.worker_list.split(',')]
    
    for i in workers:
      # restart the cluster
      FLAGS.num_workers = i
      ctx = spartan.initialize()
      
      timer = BenchTimer(i)
      util.log_info('Running benchmarks on %d workers', i)
      if FLAGS.test_optimizations:
          timer.prefix = 'opt_enabled'
          FLAGS.optimization = 1
          run_benchmarks(module, benchmarks, ctx, timer)
          
      timer.prefix = 'opt_disabled'
      FLAGS.optimization = 1
      run_benchmarks(module, benchmarks, ctx, timer)

      spartan.shutdown()
      time.sleep(1)

  if FLAGS.profile_worker:
    util.log_info('Writing worker profiles...')
    join_profiles('./_worker_profiles')


class ClusterTest(unittest.TestCase):
  '''
  Helper class for running cluster tests.
  
  Ensures a cluster instance is available before running any tests.
  '''
  @classmethod
  def setUpClass(cls):
    cls.ctx = spartan.initialize()
    
def with_ctx(fn):
  def test_fn():
      ctx = spartan.initialize()
      fn(ctx)
      
  test_fn.__name__ = fn.__name__
  return test_fn
    
 
def join_profiles(dir):
  import glob
  import pstats
  return pstats.Stats(*glob.glob(dir + '/*')).dump_stats('./worker_prof.out')

if __name__ == '__main__':
  if sys.argv[1] == 'join':
    join_profiles(sys.argv[2])
  else:
    raise Exception, 'Should not be run directly.'
