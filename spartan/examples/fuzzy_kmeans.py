import numpy as np
from spartan import expr, util
from spartan.array import extent

'''
def _calc_probability(point, centers, m):
  distances = np.square(point - centers).sum(axis=1)
  distances[distances == 0] = 0.0000000001

  pi = np.zeros(centers.shape[0])
  for i in range(centers.shape[0]):
    pi[i] = 1.0 / np.power(distances[i]/distances, 2.0 / (m - 1)).sum()

  return pi


def _fuzzy_kmeans_mapper(array, ex, old_centers, centers, counts, labels, m):
  points = array.fetch(ex)
  old_centers = old_centers[:]

  new_centers = np.zeros_like(old_centers)
  new_counts = np.zeros((old_centers.shape[0], 1))
  new_labels = np.zeros(points.shape[0], dtype=np.int)
  for i in range(points.shape[0]):
    point = points[i]
    prob = _calc_probability(point, old_centers, m)
    new_labels[i] = np.argmax(prob)

    for i in prob.nonzero()[0]:
      new_counts[i] += prob[i]
      new_centers[i] += prob[i] * point

  centers.update(extent.from_shape(centers.shape), new_centers)
  counts.update(extent.from_shape(counts.shape), new_counts)
  labels.update(extent.create((ex.ul[0],), (ex.lr[0],), labels.shape), new_labels)
  return []
'''


def fuzzy_kmeans(points, k=10, num_iter=10, m=2.0, centers=None):
  '''
  clustering data points using fuzzy kmeans clustering method.

  Args:
    points(Expr or DistArray): the input data points matrix.
    k(int): the number of clusters.
    num_iter(int): the max iterations to run.
    m(float): the parameter of fuzzy kmeans.
    centers(Expr or DistArray): the initialized centers of each cluster.
  '''
  points = expr.force(points)
  num_dim = points.shape[1]
  if centers is None:
      centers = expr.rand(k, num_dim)

  labels = expr.zeros((points.shape[0],), dtype=np.int)

  for iter in range(num_iter):
    centers = expr.as_array(centers)
    points_broadcast = expr.reshape(points, (points.shape[0], 1, points.shape[1]))
    centers_broadcast = expr.reshape(centers, (1, centers.shape[0], centers.shape[1]))
    distances = expr.sum(expr.square(points_broadcast - centers_broadcast), axis=2)
    # This is used to avoid dividing zero
    distances = distances + 0.00000000001
    util.log_info('distances shape %s' % str(distances.shape))
    distances_broadcast = expr.reshape(distances, (distances.shape[0], 1,
                                                   distances.shape[1]))
    distances_broadcast2 = expr.reshape(distances, (distances.shape[0],
                                                    distances.shape[1], 1))
    prob = 1.0 / expr.sum(expr.power(distances_broadcast / distances_broadcast2,
                                     2.0 / (m - 1)), axis=2)
    prob.force()
    counts = expr.sum(prob, axis=0)
    counts = expr.reshape(counts, (counts.shape[0], 1))
    labels = expr.argmax(prob, axis=1)
    centers = expr.sum(expr.reshape(points, (points.shape[0], 1, points.shape[1])) *
                       expr.reshape(prob, (prob.shape[0], prob.shape[1], 1)),
                       axis=0)

    # We assume that the size of centers are relative small that can be handled
    # on the master.
    counts = counts.glom()
    centers = centers.glom()
    # If any centroids don't have any points assigned to them.
    zcount_indices = (counts == 0).reshape(k)

    if np.any(zcount_indices):
      # One or more centroids may not have any points assigned to them, which results in their
      # position being the zero-vector.  We reseed these centroids with new random values
      # and set their counts to 1 in order to get rid of dividing by zero.
      counts[zcount_indices, :] = 1
      centers[zcount_indices, :] = np.random.rand(np.count_nonzero(zcount_indices),
                                                  num_dim)

    centers = centers / counts
  return labels
'''

  for iter in range(num_iter):
    new_centers = expr.ndarray((k, num_dim), reduce_fn=lambda a, b: a + b, tile_hint=(k, num_dim))
    new_counts = expr.ndarray((k, 1), dtype=np.float, reduce_fn=lambda a, b: a + b, tile_hint=(k, 1))
    expr.shuffle(points, _fuzzy_kmeans_mapper, kw={'old_centers': centers,
                                                   'centers': new_centers,
                                                   'counts': new_counts,
                                                   'labels': labels,
                                                   'm': m}).force()

    # If any centroids don't have any points assigned to them.
    zcount_indices = (new_counts.glom() == 0).reshape(k)

    if np.any(zcount_indices):
      # One or more centroids may not have any points assigned to them, which results in their
      # position being the zero-vector.  We reseed these centroids with new random values
      # and set their counts to 1 in order to get rid of dividing by zero.
      new_counts[zcount_indices, :] = 1
      new_centers[zcount_indices, :] = np.random.rand(np.count_nonzero(zcount_indices), num_dim)

    centers = new_centers / new_counts
  return labels
'''
