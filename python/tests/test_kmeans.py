from spartan import ModSharder, util, sum_accum
from spartan.array import expr
from spartan.dense import distarray, extent
import numpy as np
import spartan
import test_common

N_PTS = 10*10
N_CENTERS = 10
N_DIM = 5

distarray.TILE_SIZE = 10

# An implementation of K-means by-hand.
def min_dist(ex, tile, centers):
  #util.log('%s %s', centers.shape, tile.shape)
  dist = np.dot(centers, tile[:].T)
  min_dist = np.argmin(dist, axis=0)
#   util.log('%s %s', extent, dist.shape)
  yield extent.drop_axis(ex, 1), min_dist

def sum_centers(kernel, args):
  min_idx_id, pts_id, new_centers_id = args
  
  min_idx = kernel.table(min_idx_id)
  tgt = kernel.table(new_centers_id)
  
  c_pos = np.zeros((N_CENTERS, N_DIM))

  for shard, ex, tile in kernel.table(pts_id).iter(kernel.current_shard()):
    idx = min_idx.get(shard, extent.drop_axis(ex, 1))
    for j in range(N_CENTERS):
      c_pos[j] = np.sum(tile[idx == j], axis=0)
     
  tgt.update(0, 0, c_pos)
  
  
def test_kmeans_manual(master):
  util.log('Generating points.')
  pts = distarray.rand(master, N_PTS, N_DIM)
  centers = np.random.randn(N_CENTERS, N_DIM)
  
  util.log('Generating new centers.')
  new_centers = master.create_table(sharder=ModSharder(), combiner=None, reducer=sum_accum, selector=None)
  
  util.log('Finding closest')
  min_array = distarray.map_to_array(pts, lambda ex, tile: min_dist(ex, tile, centers))
   
  util.log('Updating clusters.')
  master.foreach_shard(min_array.table, sum_centers,
                       (min_array.id(), pts.table.id(), new_centers.id()))
  
  _, centers = spartan.fetch(new_centers)[0]
  print centers
 
def test_kmeans_expr(ctx):
  pts = expr.rand(N_PTS, N_DIM)
  centers = expr.rand(N_CENTERS, N_DIM)
   
  
if __name__ == '__main__':
  test_common.run(__file__)