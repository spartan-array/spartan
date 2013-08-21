from spartan import pytable
from spartan.array import distarray
from spartan.array.distarray import DistArray
from spartan.pytable import mod_sharder, replace_accum, util, sum_accum
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
  table = master.create_table(mod_sharder, replace_accum)
  bytes = np.ndarray((10, 10), dtype=np.double)
  for i in range(5):
    for j in range(5):
      table.update('%d%d' % (i, j), bytes)
      
  master.foreach_shard(table, get_shard_kernel, tuple())
  
  
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
  

N_PTS = 10*10
N_CENTERS = 10
DIM = 5

# An implementation of K-means by-hand.
def min_dist(extent, tile, centers):
  dist = np.dot(centers, tile[:].T)
  min_dist = np.argmin(dist, axis=0)
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
  util.log('Generating points.')
  pts = DistArray.rand(master, N_PTS, DIM)
  centers = np.random.randn(N_CENTERS, DIM)
  
  util.log('Generating new centers.')
  new_centers = master.create_table(mod_sharder, sum_accum)
  
  util.log('Finding closest')
  min_array = pts.map(min_dist, centers)
   
  util.log('Updating clusters.')
  master.foreach_shard(min_array.table, sum_centers,
                       (min_array.id(), pts.table.id(), new_centers.id()))
  
  _, centers = pytable.fetch(new_centers)[0]
  print centers
  
def test_ensure(master):
  local = np.arange(100 * 100).reshape((100, 100))
  dist = DistArray.arange(master, ((100, 100)))
   
  Assert.all_eq(dist[20:40, 1:20], 
                local[20:40, 1:20])
 
  Assert.all_eq(dist[1:2, 1:20], 
                local[1:2, 1:20])
 
  Assert.all_eq(dist[5:20, 1:20], 
                local[5:20, 1:20])
  

if __name__ == '__main__':
  test_common.run_cluster_tests(__file__)
