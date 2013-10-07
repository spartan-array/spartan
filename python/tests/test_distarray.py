from spartan import ModSharder, replace_accum, util, sum_accum
from spartan.dense import distarray, extent
from spartan.dense.distarray import DistArray
from spartan.util import Assert
import numpy as np
import spartan
import test_common

distarray.TILE_SIZE = 200
  
def get_shard_kernel(kernel, args):
  s_id = kernel.current_shard()
  t_id = kernel.current_table()
  for k, v in kernel.table(t_id).iter(s_id):
    #util.log('%s, %s', k, v)
    pass

def test_fill_array(ctx):
  table = ctx.create_table(ModSharder(), combiner=None, reducer=replace_accum, selector=None)
  bytes = np.ndarray((10, 10), dtype=np.double)
  for i in range(5):
    for j in range(5):
      table.update('%d%d' % (i, j), bytes)
      
  ctx.foreach_shard(table, get_shard_kernel, tuple())
  
  
def test_distarray_empty(ctx):
  table = ctx.create_table(ModSharder(), combiner=None, reducer=replace_accum, selector=None)
  distarray.from_table(table)
  

def map_array(k, v):
  # util.log('Extent: %s', k)
  return []
  
def test_distarray_slice(ctx):
  array = distarray.create(ctx, (200, 200))
  spartan.map_inplace(array.table, map_array)
  
def test_distarray_random(ctx):
  distarray.randn(ctx, 200, 200)
  

def test_fetch(ctx):
  N = 30
  local = np.arange(N * N, dtype=np.float).reshape((N, N))
  dist = distarray.arange(ctx, ((N, N)))
   
  Assert.all_eq(dist[0:10, 0:20], local[0:10, 0:20])
  Assert.all_eq(dist[1:2, 1:20], local[1:2, 1:20])
  Assert.all_eq(dist[5:20, 1:20], local[5:20, 1:20])
  
def test_glom(ctx):
  local = np.arange(100 * 100, dtype=np.float).reshape((100, 100))
  dist = distarray.arange(ctx, ((100, 100)))
  Assert.all_eq(local, dist[:])
  Assert.all_eq(local, dist.glom())
  
def test_locality(ctx):
  dist = distarray.arange(ctx, ((100, 100)))
  for i in range(90):
    for j in range(90):
      distarray.best_locality(dist, extent.TileExtent((i, j), (i + 10, j + 10), (100, 100)))
  

if __name__ == '__main__':
  test_common.run(__file__)
