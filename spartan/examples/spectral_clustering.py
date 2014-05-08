import numpy as np
from spartan import expr, util
from spartan.array import extent
from spartan.examples import lanczos
from spartan.examples.sklearn.cluster import KMeans

def manhattan_distance(X, Y):
  '''
  Compute the Manhattan distance between point X and Y.
  
  Args:
    X(numpy.ndarray): one point.
    Y(numpy.ndarray): anther point.
  '''
  return np.abs(X - Y).sum()

def euclidean_distance(X, Y, squared=False):
  '''
  Compute the Euclidean distance between point X and Y.
  
  Args:
    X(numpy.ndarray): one point.
    Y(numpy.ndarray): anther point.
  '''
  dist = np.square(X - Y).sum()
  return dist if squared else np.sqrt(dist)

def scaled_euclidean_distance(X, Y):
  '''
  calculate scaled Euclidean distance for point X and Y.
  scale factor c = 2 * (sigma^2), where sigma = pg. 
  p is a constant of 1.5 that is used in the EigenCuts paper.
  g is the median of the diff in each dimension of Euclidean distance.
  
  Args:
    X(numpy.ndarray): one point.
    Y(numpy.ndarray): anther point.
  '''
  diff = np.square(X, Y)
  median = np.median(diff)
  sum = diff.sum()
  return 0 if median == 0 else np.exp((-sum) / (((median * 1.5) ** 2) * 2))

def rbf_distance(X, Y, gamma=1.0):
  '''
  Compute the rbf (gaussian) kernel between point X and Y.
  
  Args:
    X(numpy.ndarray): one point.
    Y(numpy.ndarray): anther point.
  '''
  K = euclidean_distance(X, Y, squared=True) * -gamma
  return np.exp(K)

distance_methods = {'manhattan': manhattan_distance,
                    'euclidean': euclidean_distance,
                    'scaled_euclidean': scaled_euclidean_distance,
                    'rbf': rbf_distance,
                    }

def _row_similarity_mapper(array, ex, similarity_measurement):
  '''
  calculate distances for each pair of points.
  
  Args:
    array(DistArray): the input data points matrix.
    ex(Extent): region being processed.
    similarity_measurement(str): distance method used to measure similarity between two points.
  '''
  measurement = distance_methods[similarity_measurement]
  points = array.fetch(ex)
  result = np.zeros((points.shape[0], array.shape[0]))
  for other_ex in array.tiles:
    if ex == other_ex:
      other_points = points
    else:
      other_points = array.fetch(other_ex)
    
    for i in range(points.shape[0]):
      for j in range(other_points.shape[0]):
        result[i, other_ex.ul[0] + j] = measurement(points[i], other_points[j])
    
  yield extent.create((ex.ul[0], 0), (ex.lr[0], array.shape[0]), (array.shape[0], array.shape[0])), result

def _laplacian_mapper(array, ex, D):
  '''
  calculate the normalized Laplacian matrix L = D^(-0.5)AD^(-0.5)

  Args:
    array(DistArray): the adjacency matrix A.
    ex(Extent): region being processed.
    D(DistArray): the diagonal matrix which is represented as a 1-dimensional vector.
  '''
  A = array.fetch(ex)
  D_sqrt = np.sqrt(D[:])
  
  D_sqrt_zeros = (D_sqrt == 0)
  D_sqrt[D_sqrt_zeros] = 1
  D_sqrt = 1.0 / D_sqrt
  
  L = np.dot(np.dot(np.diag(D_sqrt[ex.ul[0]:ex.lr[0]]), A), np.diag(D_sqrt[ex.ul[1]:ex.lr[1]]))
  L[:, ex.ul[0]:ex.lr[0]].flat[::A.shape[0] + 1] = 1
  yield ex, L
  
def spectral_cluster(points, k=10, num_iter=10, similarity_measurement='rbf'):
  '''
  clustering data points using kmeans spectral clustering method.

  Args:
    points(Expr or DistArray): the data points to be clustered.
    k(int): the number of clusters we need to generate.
    num_iter(int): the max number of iterations that kmeans clustering method runs. 
    similarity_measurement(str): distance method used to measure similarity between two points.
  '''  
  # calculate similarity for each pair of points to generate the adjacency matrix A
  A = expr.shuffle(points, _row_similarity_mapper, kw={'similarity_measurement': similarity_measurement})
  
  num_dims = A.shape[1]
  
  # Construct the diagonal matrix D
  D = expr.sum(A, axis=1, tile_hint=(A.shape[0],))
  
  # Calculate the normalized Laplacian of the form: L = D^(-0.5)AD^(-0.5)
  L = expr.shuffle(A, _laplacian_mapper, kw={'D': D})
  
  # Perform eigen-decomposition using Lanczos solver
  overshoot = min(k * 2, num_dims) 
  d, U = lanczos.solve(L, L, overshoot, True)
  U = U[:, 0:k]
  
  # Generate initial clusters which picks rows as centers if that row contains max eigen 
  # value in that column
  init_clusters = U[np.argmax(U, axis=0)]
  
  # Run kmeans clustering with init_clusters
  kmeans = KMeans(k, num_iter)
  U = expr.from_numpy(U)
  centers, labels = kmeans.fit(U, init_clusters)
  
  return labels

