import numpy as np
import spartan
from spartan import expr, util
import parakeet
from spartan.array import distarray, extent

@util.synchronized
@parakeet.jit
def _find_closest(pts, centers):
  idxs = np.zeros(pts.shape[0])
  for i in range(pts.shape[0]):
    min_dist = 1e9
    min_idx = 0
    p = pts[i]
    for j in xrange(centers.shape[0]):
      c = centers[j]
      dist = np.sum(p ** 2 - c ** 2)
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


class KMeans(object):
  def __init__(self, n_clusters = 8,  max_iter=10):
    self.n_clusters = n_clusters
    self.max_iter = max_iter

  def fit(self, X):
    """Compute k-means clustering.

    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)
    """
    num_dim = X.shape[1]
    centers = expr.rand(self.n_clusters, num_dim)
    
    for i in range(self.max_iter):
      # Reset them to zero.
      new_centers = expr.ndarray((self.n_clusters, num_dim), reduce_fn=lambda a, b: a + b)
      new_counts = expr.ndarray((self.n_clusters, 1), reduce_fn=lambda a, b: a + b)
      
      _ = expr.shuffle(X,
                        _find_cluster_mapper,
                        kw={'d_pts' : X,
                            'old_centers' : centers,
                            'new_centers' : new_centers,
                            'new_counts' : new_counts})
      _.force()
      
      new_centers = new_centers / new_counts
      centers = new_centers
    return centers
