#!/usr/bin/env python

import os
import subprocess
import sys

sys.path += ['build/.libs', 'build/pytable']

import pytable
from pytable import mod_sharder, replace_accum, util, sum_accum
from pytable.util import Assert

try:
  import numpy as np
  import pytable.array
  from pytable.array import DistArray
except:
  pass


def fetch(table):
  out = []
  for k, v in table:
    out.append((k, v))
  return out

def test_init(master):
  table = master.create_table(mod_sharder, replace_accum)

def test_master(master):
  table = master.create_table(mod_sharder, replace_accum)
  table.update('123', '456')
  Assert.eq(table.get('123'), '456')

def put_kernel(kernel, args):
  t = kernel.table(kernel.current_table())
  t.update(kernel.current_shard(), 1)
 
def test_put_kernel(master):
  table = master.create_table(mod_sharder, replace_accum)
  master.foreach_shard(table, put_kernel, tuple())
  for i in range(10):
    Assert.eq(table.get(i), 1)
  
def get_shard_kernel(kernel, args):
  s_id = kernel.current_shard()
  t_id = kernel.current_table()
  for k, v in kernel.table(t_id).iter(s_id):
    #util.log('%s, %s', k, v)
    pass

def test_fill_array(master):
  table = master.create_table(mod_sharder, replace_accum)
  bytes = np.ndarray((1000, 1000), dtype=np.double)
  for i in range(5):
    for j in range(5):
      table.update('%d%d' % (i, j), bytes)
      
  master.foreach_shard(table, get_shard_kernel, tuple())

def copy_kernel(kernel, args):
  a, b = args
  ta = kernel.table(a)
  tb = kernel.table(b)
  
  for k, v in ta.iter(kernel.current_shard()):
    tb.update(k, v)
  
def test_copy(master):
  src = master.create_table(mod_sharder, replace_accum)
  for i in range(100):
    src.update(i, i)
    
  dst = master.create_table(mod_sharder, replace_accum)
  master.foreach_shard(src, copy_kernel, (src.id(), dst.id()))
  
  src_v = fetch(src)
  dst_v = fetch(dst)
  Assert.eq(sorted(src_v), sorted(dst_v))
  
def test_distarray_empty(master):
  table = master.create_table(mod_sharder, replace_accum)
  DistArray.from_table(table)
  

def map_array(k, v):
  util.log('Extent: %s', k)
  return []
  
def test_distarray_slice(master):
  array = DistArray.create(master, (200, 200))
  pytable.map_inplace(array.table, map_array)
  
def test_distarray_random(master):
  DistArray.randn(master, 200, 200)
  

N_PTS = 100 * 100
N_CENTERS = 100
DIM = 10

# An implementation of K-means by-hand.
def min_dist(extent, tile, centers):
  dist = np.dot(centers, tile[:].T)
  min_dist = np.argmin(dist, axis=1)
#   util.log('%s %s', extent, dist.shape)
  yield extent.drop_axis(1), min_dist

def sum_centers(kernel, args):
  min_idx_id, pts_id, new_centers_id = args
  
  min_idx = kernel.table(min_idx_id)
  tgt = kernel.table(new_centers_id)
  
  c_pos = np.zeros((N_CENTERS, DIM))

  for extent, tile in kernel.table(pts_id).iter(kernel.current_shard()):
    idx = min_idx.get(extent.drop_axis(1))
    for j in range(N_CENTERS):
      c_pos[j] = np.sum(tile[idx == j], axis=0)
      
  tgt.update(0, c_pos)
  
  
def test_kmeans(master):
  pts = DistArray.randn(master, N_PTS, DIM)
  centers = np.random.randn(N_CENTERS, DIM)
  min_array = pts.map(min_dist, centers)
  new_centers = master.create_table(mod_sharder, sum_accum)
   
  master.foreach_shard(min_array.table, sum_centers,
                       (min_array.id(), pts.table.id(), new_centers.id()))
  
#   print fetch(new_centers)


def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--with_gdb', default=False, action='store_true')
  parser.add_argument('--with_valgrind', default=False, action='store_true')
  parser.add_argument('--use_cluster', default=False, action='store_true')
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
  
  mpi_debug = flags.with_gdb and flags.use_cluster
  
  args = [ 'mpirun', '-n', '8',
           '--mca', 'btl', 'tcp,self',
           '--mca', 'btl_tcp_if_include', '216.165.108.0/24',
           '-x', 'PYTHONPATH',
        ]
  
  if mpi_debug:
    args += ['-d', '--mca', 'pls_rsh_agent', 'ssh -X -n', ]
  
  if flags.use_cluster:
    args += [ '-hostfile', 'conf/mpi-cluster' ]
  else: 
    args += [ '-hostfile', 'conf/mpi-local', ]
   
  if flags.with_gdb or flags.with_valgrind:
    args += [ 'xterm', '-e', ]
  
  for t in tests:
    if flags.with_gdb:
      proc_args = args + [ 'gdb -ex run --args %s tests/test_pytable.py --run_test=%s' % (sys.executable, t) ]
    elif flags.with_valgrind:
      proc_args = args + ['valgrind %s src/tests/test_python.py --run_test=%s' % (sys.executable, t)]
    else:
      proc_args = args + [sys.executable, __file__, '--run_test=%s' % t]

    p = subprocess.Popen(env = env,
                         args=proc_args)
      
    if p.wait() != 0:
      util.log('Test %s FAILED', t)
  
if __name__ == '__main__':
  main()
