import numpy as np
import spartan
from spartan import expr, util
import parakeet
from spartan.array import distarray, extent

class Centroid(object):
  '''
  A centroid records a center point of a cluster and the number of 
  points that are contained within it.
  
  Args:
    center(numpy.array): the center point of this cluster.
    weight(float): the number of points that are contained within it.
  '''
  def __init__(self, point, weight):
    self.center = point
    self.weight = weight
    
  def __reduce__(self):
    return (Centroid, (self.center, self.weight))
  
  def __repr__(self):
    return 'Centroid(%s, %s)' % (self.center, self.weight)
  
  def update(self, other_point, weight):
    self.center = (self.center * self.weight + other_point * weight) / (self.weight + weight)
    self.weight += weight
  
  def get_center(self):
    return self.center
   
  def get_weight(self):
    return self.weight

  def set_weight(self, weight):
    self.weight = weight
     
def _estimate_distance_cutoff(points):
  '''
  estimate the distance cutoff between two points for forming new clusters. 
  It is the minimum distance of each pair of points. 
  
  Args:
    points(numpy.ndarray): points to be calculated. 
  '''
  min_distance = 1e9
  for i in range(points.shape[0]):
    for j in range(i + 1, points.shape[0]):
      dist = np.square(points[i] - points[j]).sum()
      if dist < min_distance: min_distance = dist
  return min_distance

def _find_closest_centroid(point, centroids, diff_from_point=False):
  '''
  find the closest centroid from the centroids for a point.
  Return the centroid index and the minimum distance.
  
  Args:
    point(numpy.ndarray): point to be calculated.
    centroids(list): list of Centroids.
    diff_from_point: whether need to exclude the centroid when it is equal to the point.
  '''
  min_distance = 1e9
  min_idx = -1
  for j in range(len(centroids)):
    dist = np.square(centroids[j].get_center() - point).sum()
    if dist < min_distance and (not diff_from_point or 
                                np.any(np.not_equal(centroids[j].get_center(), point))):
      min_distance = dist
      min_idx = j
  return min_idx, min_distance

def _collapse_clusters(centroids, estimated_distance_cutoff):
  '''
  collapse the clusters when the number of clusters becomes too large.
  
  Args:
    centroids(list): list of current centroids.
    estimated_distance_cutoff(float): distance cutoff between two points for forming new clusters.
  '''
  np.random.shuffle(centroids)
  
  new_centroids = [centroids[0]]
  for i in range(1, len(centroids)):
    centroid = centroids[i]
    
    min_idx, min_distance = _find_closest_centroid(centroid.get_center(), new_centroids)
        
    sample = np.random.rand()
    if sample < centroid.get_weight() * min_distance / estimated_distance_cutoff:
      new_centroids.append(centroid)
    else:
      new_centroids[min_idx].update(centroid.get_center(), centroid.get_weight())
   
  return new_centroids
        
def _streaming_mapper(array, ex, k):
  '''
  clustering the local points online.
  
  Args:
    array(DistArray): points to be clustered.
    ex(Extent): region being processed.
    k(int): the number of final clusters.
  '''
  points = array.fetch(ex)
  
  estimated_clusters = int(k * np.log(array.shape[0]))
  estimated_distance_cutoff = _estimate_distance_cutoff(points[:min(1000, points.shape[0])])

  centroids = [Centroid(points[0], 1.0)]
  num_processed_points = 1
  
  for i in range(1, points.shape[0]):
    min_idx, min_distance = _find_closest_centroid(points[i], centroids)
 
    sample = np.random.rand()
    if sample < min_distance / estimated_distance_cutoff:
      centroids.append(Centroid(points[i], 1.0))
    else:
      centroids[min_idx].update(points[i], 1.0)
    
    num_processed_points += 1
    
    # when the number of the clusters becomes too large, 
    # collapse clusters and increase the threshold.
    if len(centroids) > 2 * estimated_clusters:      
      np.random.shuffle(centroids)
      centroids = _collapse_clusters(centroids, estimated_distance_cutoff)
      
      estimated_clusters = int(max(estimated_clusters, 20 * np.log(num_processed_points)))
      if len(centroids) > estimated_clusters:
        estimated_distance_cutoff *= 1.3
      
  yield None, centroids

def _cluster_mapper(array, ex, centers):
  '''
  label the cluster id for each data point.
  
  Args:
    array(DistArray): the input data points matrix.
    ex(Extent): region being processed.
    centers(numpy.array): the center points for each cluster.
  '''
  points = array.fetch(ex)
  labels = np.zeros(points.shape[0], dtype=np.int32)
  for i in range(points.shape[0]):
    point = points[i]
    max = -1
    max_id = -1
    for j in range(centers.shape[0]):
      dist = np.square(centers[j] - point).sum()
      pdf = 1.0 / (1 + dist)
      if max < pdf:
        max = pdf
        max_id = j
        
    labels[i] = max_id
    
  yield extent.create((ex.ul[0],), (ex.lr[0],), (array.shape[0],)), labels


def _iterative_assign_points(data_points, centroids, k, num_iters, trim_factor, correct_weights):
  '''
  update cluster centers to be the centroid of the nearest data points.  

  Args:
    data_points(list): list of points to be assigned.
    centroids(list): list of cluster centroids.
    k(int): the final number of clusters.
    num_iters(int): the number of iterations to run in each ball kmeans run.
    trim_factor(float): the ball kmeans parameter to separate the nearest points and distant points.
    correct_weights(bool): whether to correct the weights of the centroids.
  '''
  closest_cluster_distances = np.zeros(k)
  assignments = np.ones(len(data_points)) * -1
  
  for i in range(num_iters):
    changed = False

    # compute the minimum distance between each cluster
    for j in range(k):
      min_idx, min_distance =  _find_closest_centroid(centroids[j].get_center(), centroids, True) 
      closest_cluster_distances[j] = min_distance
    
    new_centroids = [Centroid(centroid.get_center(), 0) for centroid in centroids]
    
    for j in range(len(data_points)):
      point = data_points[j]
      min_idx, min_distance = _find_closest_centroid(point.get_center(), centroids, False)  
      
      # update its cluster assignment if necessary.
      if min_idx != assignments[j]:
          changed = True
          assignments[j] = min_idx
    
      # only update if the point is near enough.
      if min_distance < trim_factor * closest_cluster_distances[min_idx]:
        new_centroids[min_idx].update(point.get_center(), point.get_weight())
    
    centroids = new_centroids

    if not changed:
        break
    
  if correct_weights:
    for centroid in centroids: centroid.set_weight(0)
    
    for point in data_points:
      min_idx, min_distance = _find_closest_centroid(point.get_center(), centroids, False)
      centroids[min_idx].set_weight(centroids[min_idx].get_weight() + point.get_weight())
      
  return centroids
  
def ball_kmeans(data_points, k, num_iters, num_runs, trim_factor, test_probability, correct_weights):  
  '''
  ball kmeans clustering algorithm.  

  Args:
    data_points(list): list of points to be assigned.
    k(int): the final number of clusters.
    num_iters(int): the number of iterations to run in each ball kmeans run.
    num_runs(int): the number of ball kmeans to run.
    trim_factor(float): the ball kmeans parameter to separate the nearest points and distant points.
    test_probability(float): the percentage of points to be chosen as test set.
    correct_weights(bool): whether to correct the weights of the centroids.
  '''
  num_test = int(np.ceil(test_probability * len(data_points)))
  train_set = data_points[num_test:]
  test_set = data_points[0:num_test]
  
  prob = np.array([point.get_weight() for point in train_set])
  prob = prob / prob.sum()
  
  best_cost = 1e9
  best_centroids = None
  
  for i in range(num_runs):
    centroids = np.random.choice(train_set, size=k, replace=False, p=prob)

    if num_runs > 1:
      centroids = _iterative_assign_points(train_set, centroids, k, num_iters, 
                                           trim_factor, correct_weights)

      cost = 0.0
      for point in test_set:
        min_idx, min_distance = _find_closest_centroid(point.get_center(), centroids, False)
        cost += min_distance
      
      if cost < best_cost:
        best_cost = cost
        best_centroids = centroids
    else:
      return _iterative_assign_points(data_points, centroids, k, num_iters, 
                                      trim_factor, correct_weights)
  
  if correct_weights:
    for point in test_set:
      min_idx, min_distance = _find_closest_centroid(point.get_center(), best_centroids, False)
      best_centroids[min_idx].set_weight(best_centroids[min_idx].get_weight() + point.get_weight())
  
  return best_centroids
  
def streaming_kmeans(points, k=10, num_iters=10, num_ballkmeans_runs=2, trim_factor=0.9, 
                     test_probability=0.1, correct_weight=False):
  '''
  clustering data points using streaming kmeans method.  

  Args:
    points(DistArray): data points to be clustered.
    k(int): the final number of clusters.
    num_iters(int): the number of iterations to run in each ball kmeans run.
    num_ballkmeans_runs(int): the number of ball kmeans to run.
    trim_factor(float): the ball kmeans parameter to separate the nearest points and distant points.
    test_probability(float): the percentage of points to be chosen as test set.
    correct_weights(bool): whether to correct the weights of the centroids.
  '''  
  centroids = expr.tile_operation(points, 
                                  _streaming_mapper, 
                                  kw={'k': k}).force()
  
  new_centroids = []
  for tile_result in centroids.values():
    for centroids_list in tile_result:
      new_centroids.extend(centroids_list)
  
  centriods = ball_kmeans(new_centroids, k, num_iters, num_ballkmeans_runs, trim_factor, 
                          test_probability, correct_weight)
  
  centers = np.zeros((k, points.shape[1]))
  for i in range(k):
    centers[i] = centriods[i].get_center()

  return expr.shuffle(points, _cluster_mapper, kw={'centers': centers})
  
  
  
