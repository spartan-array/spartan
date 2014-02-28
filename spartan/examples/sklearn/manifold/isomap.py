"""isomap for manifold learning"""
import numpy as np
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
#from ..utils import check_arrays
from ..util.graph_shortest_path import graph_shortest_path
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import KernelCenterer
from spartan import expr, blob_ctx, core
from spartan.array import extent

def _shortest_path_mapper(ex,
                          kng,
                          directed,
                          dist_matrix):
  """
  Mapper kernel for finding shortest path for a subset of points.  
  
  kng is supposed to be a sparse matrix which represents the distance among each pair
  of points.
  
  dist_matrix is the target matrix which we need to fill with the shortest path between
  each pair of points.

  Each kernel is responsible for finding the shortests path among a subset of points.
  """
  row_beg = ex.ul[0]
  row_end = ex.lr[0]
  
  local_dist_matrix = graph_shortest_path(kng,
                                          row_beg,
                                          row_end,
                                          directed=directed)
  '''
  local_dist_matrix is a NxN matrix where the M(i,j) is the shortest
  path between i and j if it's positive, otherwise it's zero.  
  '''
  dist_matrix.update(extent.from_shape(local_dist_matrix.shape), 
                      local_dist_matrix)

  result = core.LocalKernelResult()
  return result

class Isomap(object):
  """Isomap Embedding

  Non-linear dimensionality reduction through Isometric Mapping

  Parameters
  ----------
  n_neighbors : integer
      number of neighbors to consider for each point.

  n_components : integer
      number of coordinates for the manifold

  eigen_solver : ['auto'|'arpack'|'dense']
      'auto' : Attempt to choose the most efficient solver
          for the given problem.
      'arpack' : Use Arnoldi decomposition to find the eigenvalues
          and eigenvectors.
      'dense' : Use a direct solver (i.e. LAPACK)
          for the eigenvalue decomposition.

  tol : float
      Convergence tolerance passed to arpack or lobpcg.
      not used if eigen_solver == 'dense'.

  max_iter : integer
      Maximum number of iterations for the arpack solver.
      not used if eigen_solver == 'dense'.

  neighbors_algorithm : string ['auto'|'brute'|'kd_tree'|'ball_tree']
      Algorithm to use for nearest neighbors search,
      passed to neighbors.NearestNeighbors instance.

  Attributes
  ----------
  `embedding_` : array-like, shape (n_samples, n_components)
      Stores the embedding vectors.

  `kernel_pca_` : object
      `KernelPCA` object used to implement the embedding.

  `training_data_` : array-like, shape (n_samples, n_features)
      Stores the training data.

  `nbrs_` : sklearn.neighbors.NearestNeighbors instance
      Stores nearest neighbors instance, including BallTree or KDtree
      if applicable.

  `dist_matrix_` : array-like, shape (n_samples, n_samples)
      Stores the geodesic distance matrix of training data.

  References
  ----------

  [1] Tenenbaum, J.B.; De Silva, V.; & Langford, J.C. A global geometric
      framework for nonlinear dimensionality reduction. Science 290 (5500)
  """
  def __init__(self, n_neighbors=5, n_components=2, eigen_solver='auto',
               tol=0, max_iter=None, 
               neighbors_algorithm='auto'):

      self.n_neighbors = n_neighbors
      self.n_components = n_components
      self.eigen_solver = eigen_solver
      self.tol = tol
      self.max_iter = max_iter
      self.neighbors_algorithm = neighbors_algorithm
      self.nbrs_ = NearestNeighbors(n_neighbors=n_neighbors,
                                    algorithm=neighbors_algorithm)
  
  def _fit_transform(self, X):
    self.nbrs_.fit(X)
    self.training_data_ = self.nbrs_._fit_X 
    self.kernel_pca_ = KernelPCA(n_components=self.n_components,
                                  kernel="precomputed",
                                  eigen_solver=self.eigen_solver,
                                  tol=self.tol, max_iter=self.max_iter)
    
    kng = kneighbors_graph(self.nbrs_, self.n_neighbors, mode="distance")
    n_points = X.shape[0]
    n_workers = blob_ctx.get().num_workers

    if n_points < n_workers:
      tile_hint = (1, )
    else:
      tile_hint = (n_points / n_workers, )

    """
    task_array is used for deciding the idx of starting points and idx of endding points 
    that each tile needs to find the shortest path among.
    """
    task_array = expr.ndarray((n_points,), tile_hint=tile_hint)
    task_array = task_array.force()
    
    #dist matrix is used to hold the result
    dist_matrix = expr.ndarray((n_points, n_points), reduce_fn=lambda a,b:a+b).force()
    results = task_array.foreach_tile(mapper_fn = _shortest_path_mapper,
                                      kw = {'kng' : kng,
                                            'directed' : False,
                                            'dist_matrix' : dist_matrix})
    self.dist_matrix_ = dist_matrix.glom()
    G = self.dist_matrix_ ** 2
    G *= -0.5
    self.embedding_ = self.kernel_pca_.fit_transform(G)

  def fit(self, X):
    """
    Fit the model from data in X and transform X.
    Parameters
    ----------
    X: {array-like, sparse matrix, BallTree, KDTree}
        Training vector, where n_samples in the number of samples
        and n_features is the number of features.

    Returns
    -------
    X_new: array-like, shape (n_samples, n_components)
    """
    self._fit_transform(X)
    return self
