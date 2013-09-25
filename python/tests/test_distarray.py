import spartan
from spartan.array import distarray
from spartan.array.distarray import DistArray
from spartan import ModSharder, replace_accum, util, sum_accum
from spartan.util import Assert
import numpy as np
import test_common

distarray.TILE_SIZE = 10
  
def get_shard_kernel(kernel, args):
  s_id = kernel.current_shard()
  t_id = kernel.current_table()
  for k, v in kernel.table(t_id).iter(s_id):
    #util.log('%s, %s', k, v)
    pass

def test_fill_array(master):
  table = master.create_table(ModSharder(), combiner=None, reducer=replace_accum, selector=None)
  bytes = np.ndarray((10, 10), dtype=np.double)
  for i in range(5):
    for j in range(5):
      table.update('%d%d' % (i, j), bytes)
      
  master.foreach_shard(table, get_shard_kernel, tuple())
  
  
def test_distarray_empty(master):
  table = master.create_table(ModSharder(), combiner=None, reducer=replace_accum, selector=None)
  distarray.from_table(table)
  

def map_array(k, v):
  # util.log('Extent: %s', k)
  return []
  
def test_distarray_slice(master):
  array = distarray.create(master, (200, 200))
  spartan.map_inplace(array.table, map_array)
  
def test_distarray_random(master):
  distarray.randn(master, 200, 200)
  

def test_ensure(master):
  local = np.arange(100 * 100, dtype=np.float).reshape((100, 100))
  dist = distarray.arange(master, ((100, 100)))
   
  Assert.all_eq(dist[0:10, 0:20], local[0:10, 0:20])
  Assert.all_eq(dist[1:2, 1:20], local[1:2, 1:20])
  Assert.all_eq(dist[5:20, 1:20], local[5:20, 1:20])
  

if __name__ == '__main__':
  test_common.run_cluster_tests(__file__)
