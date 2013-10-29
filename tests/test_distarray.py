import numpy as np
from spartan import ModSharder, replace_accum, util, sum_accum
import spartan
from spartan.dense import distarray, extent
from spartan.dense.distarray import DistArray
from spartan.util import Assert
from spartan.wrap import get_master, _mapper_kernel
import test_common


def fetch(table):
  out = []
  for s, k, v in table:
    out.append((k, v))
  return out


def map_inplace(table, fn, kw):
  src = table
  dst = src
  master = get_master()
  master.foreach_shard(table, _mapper_kernel, 
                       (src.id(), dst.id(), fn, kw))
  return dst


def create_with(master, shape, init_fn):
  d = distarray.create(master, shape)
  map_inplace(d.table, init_fn, kw={})
  return d 

def _create_rand(extent, data):
  data[:] = np.random.rand(*extent.shape)

def _create_randn(extent, data):
  data[:] = np.random.randn(*extent.shape)
  
def _create_ones(extent, data):
  util.log_info('Updating %s, %s', extent, data)
  data[:] = 1

def _create_zeros(extent, data):
  data[:] = 0

def _create_range(ex, data):
  Assert.eq(ex.shape, data.shape)
  pos = extent.ravelled_pos(ex.ul, ex.array_shape)
  sz = np.prod(ex.shape)
  data[:] = np.arange(pos, pos+sz).reshape(ex.shape)
  
def randn(master, *shape):
  return create_with(master, shape, _create_randn)

def rand(master, *shape):
  return create_with(master, shape, _create_rand)

def ones(master, shape):
  return create_with(master, shape, _create_ones)

def zeros(master, shape):
  return create_with(master, shape, _create_zeros)
  
def arange(master, shape):
  return create_with(master, shape, _create_range)

  
def get_shard_kernel(kernel, args):
  s_id = kernel.current_shard()
  t_id = kernel.current_table()
  for s, k, v in kernel.table(t_id).iter(s_id):
    #util.log_info('%s, %s', k, v)
    pass

def map_array(k, v):
  # util.log_info('Extent: %s', k)
  return []

# helper to simplify fetching
# DistArray purposefully does not export the getitem interface.
class Fetcher(object):
  def __init__(self, darray):  
    self.darray = darray
    
  def __getitem__(self, idx):
    ex = extent.from_slice(idx, self.darray.shape)
    return self.darray.fetch(ex)

class TestDistarray(test_common.ClusterTest):
  TILE_SIZE = 100
  
  def test_fill_array(self):
    table = self.ctx.create_table(ModSharder(), combiner=None, reducer=replace_accum, selector=None)
    bytes = np.ndarray((10, 10), dtype=np.double)
    for i in range(5):
      for j in range(5):
        table.update(0, '%d%d' % (i, j), bytes)
        
    self.ctx.foreach_shard(table, get_shard_kernel, tuple())
    
  
  def test_distarray_empty(self):
    table = self.ctx.create_table(ModSharder(), combiner=None, reducer=replace_accum, selector=None)
    distarray.from_table(table)
    
  
  
  def test_distarray_slice(self):
    array = distarray.create(self.ctx, (200, 200))
    map_inplace(array.table, map_array, {})
    
  
  def test_random(self):
    randn(self.ctx, 200, 200)
  
  
  def test_fetch(self):
    N = 30
    local = np.arange(N * N, dtype=np.float).reshape((N, N))
    dist = arange(self.ctx, ((N, N)))
    fetch = Fetcher(dist)
    Assert.all_eq(fetch[0:10, 0:20], local[0:10, 0:20])
    Assert.all_eq(fetch[1:2, 1:20], local[1:2, 1:20])
    Assert.all_eq(fetch[5:20, 1:20], local[5:20, 1:20])
    
  
  def test_glom(self):
    local = np.arange(100 * 100, dtype=np.float).reshape((100, 100))
    dist = arange(self.ctx, ((100, 100)))
    fetch = Fetcher(dist)
    Assert.all_eq(local, fetch[:])
    Assert.all_eq(local, dist.glom())
    
  
  def test_locality(self):
    dist = arange(self.ctx, ((100, 100)))
    for i in range(0, 100, 10):
      for j in range(0, 100, 10):
        distarray.best_locality(dist, extent.create((i, j), (i + 10, j + 10), (100, 100)))
    
