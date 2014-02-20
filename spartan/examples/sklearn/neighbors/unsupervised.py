import numpy as np
from spartan import expr, core, array
from sklearn.neighbors import NearestNeighbors as SKNN

def _knn_mapper(ex,
                X,
                Q,
                n_neighbors,
                algorithm):
  """
  knn kernel for finding k(n_neighbors) nearest neighbors of a giving query set(Q).
  
  Each kernel call finds k nearest neighbors of subset(one tile) of X. Then we sends
  the KNN candidates to master to find out the real KNN.

  Parameters
  ----------
  X : array. The search set. We'll try to find KNN of Q set in one tile of X.
  
  Q : array. Query set, we'll try to find its KNN candidates in X.

  n_neighbors : integer. Specify the K.

  algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
      Algorithm used to compute the nearest neighbors locally:
  """
  result = core.LocalKernelResult()
  row_start = ex.ul[0]
  col_start = ex.ul[1]
 
  """
  If it's not started with the first column, we skip. 
  Let the tile starts with first column does the computation. 
  """
  if col_start != 0:
    return result
  
  ul = ex.ul
  lr = (ex.lr[0], X.shape[1])
  ex = array.extent.create(ul, lr, ex.array_shape)
  X = X.fetch(ex)
  Q = Q.glom()
  
  """ Run sklearn SKK locally to find the KNN candidates """ 
  nbrs = SKNN(n_neighbors=n_neighbors, 
                algorithm=algorithm).fit(X)

  dist, ind = nbrs.kneighbors(Q)
  ind += row_start
  result = core.LocalKernelResult()
  result.result = (dist, ind)
  return result


class NearestNeighbors(object):
  """ 
  Unsupervised learner for implementing neighbor searches.

    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`k_neighbors` queries.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDtree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    Examples
    --------
      >>> from spartan.examples.sklearn.neighbors import NearestNeighbors
      >>> samples = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]

      >>> neigh = NearestNeighbors(2, 0.4)
      >>> neigh.fit(samples)  #doctest: +ELLIPSIS
      NearestNeighbors(...)

      >>> neigh.kneighbors([[0, 0, 1.3]], 2)
      ... #doctest: +ELLIPSIS
      array([[2, 0]]...)

    Notes
    -----
    http://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm

  """
  def __init__(self,
                n_neighbors=5,
                algorithm='auto'):
    self.n_neighbors = n_neighbors
    self.algorithm = algorithm

  def fit(self, X):
    if isinstance(X, np.ndarray):
      X = expr.from_numpy(X)    
    if isinstance(X, expr.Expr):
      X = X.force()

    self.X = X
    return self

  def kneighbors(self, X, n_neighbors=None):
    """Finds the K-neighbors of a point.

        Returns distance

        Parameters
        ----------
        X : array-like, last dimension same as that of fit data
            The new point.

        n_neighbors : int
            Number of neighbors to get (default is the value
            passed to the constructor).

        Returns
        -------
        dist : array
            Array representing the lengths to point, only present if
            return_distance=True

        ind : array
            Indices of the nearest points in the population matrix.
    """
    if isinstance(X, np.ndarray):
      X = expr.from_numpy(X)
    
    if isinstance(X, expr.Expr):
      X = X.force()

    results = self.X.foreach_tile(mapper_fn = _knn_mapper,
                                  kw = {'X' : self.X, 'Q' : X,
                                        'n_neighbors' : self.n_neighbors,
                                        'algorithm' : self.algorithm})
    dist = None
    ind = None
    """ Get the KNN candidates for each tile of X, then find out the real KNN """
    for k, v in results.iteritems():
      if dist is None:
        dist = v[0]
        ind = v[1]
      else:
        dist = np.concatenate((dist, v[0]), axis=1)
        ind = np.concatenate((ind, v[1]), axis=1)

    mask = np.argsort(dist, axis=1)[:, :self.n_neighbors]
    new_dist = np.array([dist[i][mask[i]] for i, r in enumerate(dist)])
    new_ind = np.array([ind[i][mask[i]] for i, r in enumerate(ind)]) 
    return new_dist, new_ind
