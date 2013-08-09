from pytable import mod_sharder, replace_accum, util, sum_accum
from pytable.array.distarray import DistArray
import numpy as np
import pytable
import test_common
  
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
  

N_PTS = 1000 * 1000 * 1000
N_CENTERS = 100
DIM = 3

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

if __name__ == '__main__':
  test_common.run_cluster_tests(__file__)
