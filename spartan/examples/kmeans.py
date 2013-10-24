import numpy as np
from spartan import expr, util
from spartan.dense import distarray, extent
from spartan.expr import lazify
import spartan
from spartan.util import divup

import parakeet

@util.synchronized
@parakeet.jit
def _find_closest(pts, centers):
  idxs = np.zeros(pts.shape[0])

  for i in range(pts.shape[0]):
    min_dist = 1e9
    min_idx = 0
    p = pts[i]
    for j in range(len(centers.shape)):
      c = centers[j]
      dist = np.dot(p, c) 
      if dist < min_dist:
        min_dist = dist
        min_idx = j
    
    idxs[i] = min_idx
  return idxs


def _find_cluster_mapper(inputs, ex, d_pts, old_centers, 
                         new_centers, new_counts):
  #util.log_info('Mapping...')
  centers = old_centers.glom()
  pts = d_pts.fetch(ex)

  closest = _find_closest(pts, centers)
  
  #dists = np.dot(pts, centers.T)
  
  # assign points to nearest centroid
  #closest = np.argmin(dists, axis=1)
  
  l_counts = np.zeros((centers.shape[0], 1))
  l_centers = np.zeros_like(centers)
  
  for i in range(centers.shape[0]):
    matching = closest == i
    l_counts[i,0] = matching.sum()
    l_centers[i] = pts[matching].sum(axis=0)
    
  # update centroid positions
  new_centers.update(extent.from_shape(new_centers.shape), l_centers)
  new_counts.update(extent.from_shape(new_counts.shape), l_counts)
  
def run(num_pts, num_centers, num_dim):
  ctx = spartan.get_master()
  
  pts = expr.rand(num_pts, num_dim,
                  tile_hint=(divup(num_pts, ctx.num_workers()), num_dim)).force()
                             
  centers = expr.rand(num_centers, num_dim).force()
  
  #util.log_info('%s', centers.glom())

  new_centers = expr.ndarray((num_centers, num_dim)).force()
  new_counts = expr.ndarray((num_centers,1)).force()
  
  for i in range(10):
    _ = expr.map_extents(pts, 
                         _find_cluster_mapper,
                         kw={'d_pts' : pts, 
                             'old_centers' : centers,
                             'new_centers' : new_centers, 
                             'new_counts' : new_counts })
    _.force()
    
    new_centers = lazify(new_centers) / lazify(new_counts)
    new_centers = new_centers.force()
