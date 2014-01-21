import numpy as np
from spartan import expr, util
from spartan.array import distarray, extent
from spartan.expr import lazify, eager
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
  centers = old_centers.glom()
  pts = d_pts.fetch(ex)

  closest = _find_closest(pts, centers)
  
  l_counts = np.zeros((centers.shape[0], 1))
  l_centers = np.zeros_like(centers)
  
  for i in range(centers.shape[0]):
    matching = closest == i
    l_counts[i,0] = matching.sum()
    l_centers[i] = pts[matching].sum(axis=0)
    
  # update centroid positions
  new_centers.update(extent.from_shape(new_centers.shape), l_centers)
  new_counts.update(extent.from_shape(new_counts.shape), l_counts)
  return []
  
def run(num_pts, num_centers, num_dim):
  ctx = spartan.blob_ctx.get()

  pts = expr.rand(num_pts, num_dim,
                  tile_hint=(divup(num_pts, ctx.num_workers), num_dim)).force()
                             
  centers = expr.rand(num_centers, num_dim).force()
  
  #util.log_info('%s', centers.glom())

  new_centers = expr.ndarray((num_centers, num_dim))
  new_counts = expr.ndarray((num_centers,1))
  
  for i in range(10):
    _ = expr.shuffle(pts, 
                     _find_cluster_mapper,
                     kw={'d_pts' : pts,
                         'old_centers' : centers,
                         'new_centers' : new_centers,
                         'new_counts' : new_counts })
    _.force()
    
    new_centers = expr.eager(new_centers / new_counts)
