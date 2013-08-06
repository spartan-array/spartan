#!/usr/bin/env python

from test_common import Assert
import cPickle
import numpy as np
import os
import subprocess
import sys

sys.path += ['build/.libs', 'build/pytable']


try:
  import pytable
  import pytable.array
  from pytable.array import DistArray
  from pytable import mod_sharder, replace_accum, util
except ImportError, e:
  print 'Skipping sparrow import.', e 

def fetch(table):
  out = []
  it = table.get_iterator()
  while not it.done():
    out.append((it.key(), it.value()))
    it.next()
  return out

def test_init(master):
  table = pytable.create_table(master, mod_sharder, replace_accum)

def test_master(master):
  table = pytable.create_table(master, mod_sharder, replace_accum)
  table.update('123', '456')
  Assert.eq(table.get('123'), '456')

def put_kernel(kernel, args):
  kernel = pytable.as_kernel(kernel)
  t = pytable.get_table(kernel, kernel.table_id())
  t.update(kernel.shard_id(), 1)
 
def test_put_kernel(master):
  table = pytable.create_table(master, mod_sharder, replace_accum)
  pytable.foreach_shard(master, table, put_kernel, tuple())
  for i in range(10):
    Assert.eq(table.get(i), 1)
  
def get_shard_kernel(k_id, args):
  kernel = pytable.as_kernel(k_id)
  s_id = kernel.shard_id()
  t_id = kernel.table_id()
  it = pytable.get_table(kernel, t_id).get_iterator(s_id)
  
  while not it.done():
    util.log('%s, %s', it.key(), len(it.value()))
    it.next()

def test_fill_array(master):
  table = pytable.create_table(master, mod_sharder, replace_accum)
  bytes = np.ndarray((1000, 1000), dtype=np.double)
  for i in range(5):
    for j in range(5):
      table.update('%d%d' % (i, j), bytes)
      
  pytable.foreach_shard(master, table, get_shard_kernel, tuple())

def copy_kernel(k_id, args):
  kernel = pytable.as_kernel(k_id)
  a, b = args
  ta = pytable.get_table(kernel, a)
  tb = pytable.get_table(kernel, b)
  
  it = ta.get_iterator(kernel.shard_id())
  while not it.done():
    tb.update(it.key(), it.value())
    it.next()
  
def test_copy(master):
  src = pytable.create_table(master, mod_sharder, replace_accum)
  for i in range(100):
    src.update(i, i)
    
  dst = pytable.create_table(master, mod_sharder, replace_accum)
  pytable.foreach_shard(master, src, copy_kernel, (src.id(), dst.id()))
  
  src_v = fetch(src)
  dst_v = fetch(dst)
  Assert.eq(sorted(src_v), sorted(dst_v))
  
def test_distarray_empty(master):
  table = pytable.create_table(master, mod_sharder, replace_accum)
  empty = DistArray.from_table(table)

def map_array(k, v):
  util.log('Extent: %s', k)
  return []
  
def test_distarray_slice(master):
  array = DistArray.create(master, (200, 200))
  pytable.map_inplace(array.table, map_array)
  
def test_distarray_random(master):
  array = DistArray.randn(master, 200, 200)
  
# def test_distarray_add(master):
#   a = DistArray.ones(master, 1000, 1000)
#   b = DistArray.ones(master, 1000, 1000)
#   a + b

def min_dist_kernel(extent, tile, centers):
  dist = np.dot(centers, tile[:].T)
  min_dist = np.argmin(dist, axis=1)
#   util.log('%s %s', extent, dist.shape)
  yield extent.drop_axis(1), min_dist

def sum_centers_kernel(extent, tile):
  pass  

def test_kmeans(master):
  N_PTS = 1000000
  N_CENTERS = 100
  DIM = 10
  pts = DistArray.randn(master, N_PTS, DIM)
  centers = np.random.randn(N_CENTERS, DIM)
  min_table = pytable.map_items(pts.table, min_dist_kernel, centers)
  min_array = DistArray.from_table(min_table)
  util.log('%s', min_array.shape)
  
  

def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--run_test', default='')
  flags, argv = parser.parse_known_args()
  
  if flags.run_test:
    argv = ['python'] + argv
    master = pytable.init(argv)
    print >> sys.stderr, 'RUNNING: ', flags.run_test
    fn = eval(flags.run_test)
    fn(master)
    return
  
  env = dict(os.environ)
  env['PYTHONPATH'] += ':build/.libs/:build/pytable'
  
  # setup environment, and call MPI for each test
  tests = [k for k in globals().keys() if k.startswith('test_')]
  #tests = ['test_init']
  for t in tests:
    p = subprocess.Popen(env = env,
                         args=[
      'mpirun', '-x', 'PYTHONPATH', '-n', '2', '-hostfile', 'conf/mpi-local',
      sys.executable, __file__, '--run_test=%s' % t
      #'xterm', '-e',
      #'valgrind %s src/tests/test_python.py --run_test=%s' % (sys.executable, t)
      #'gdb -ex run --args %s tests/test_python.py --run_test=%s' % (sys.executable, t)
      ]
      )
    if p.wait() != 0:
      sys.exit(1)
  
if __name__ == '__main__':
  main()
