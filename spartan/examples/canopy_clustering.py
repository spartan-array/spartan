import numpy as np
from spartan import expr, util
from spartan.array import extent

class Canopy(object):
  '''
  A canopy records a center point and the number of points that are contained within it.
  
  Args:
    center(numpy.array): the center point of this canopy.
  '''
  def __init__(self, center):
    self.center = center
    self.num_observations = 1
    self.sum = center.copy()
  
  def observe(self, x):
    self.num_observations += 1
    self.sum += x
    
  def get_center(self):
    return self.center
  
  def get_num_observations(self):
    return self.num_observations
  
  def compute_parameters(self):
    if self.num_observations > 0:
      self.center = self.sum / self.num_observations 
      
def _canopy_mapper(array, ex, t1, t2, cf):
  '''
  find all the canopies for the local data points.
  
  Args:
    array(DistArray): the input data points matrix.
    ex(Extent): region being processed.
    t1(float): distance threshold between center point and the points within a canopy. 
    t2(float): distance threshold between center point and the points within a canopy.
    cf(int): the minimum canopy size.
  '''
  points = array.fetch(ex)
  canopies = []
  for i in range(points.shape[0]):
    point = points[i]
    point_strongly_bound = False
    
    for c in canopies:
      dist = np.square(c.get_center() - point).sum()
      if dist < t1: c.observe(point)
      point_strongly_bound |= (dist < t2)
      
    if not point_strongly_bound:
      canopies.append(Canopy(point))
  
  result = []
  for i in range(len(canopies)):
    c = canopies[i]
    c.compute_parameters()
    if c.get_num_observations() > cf:
      result.append(c.get_center())

  yield None, np.array(result)

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

def find_centers(point_blocks, t1, t2, cf):
  '''
  find the final center points.
  
  Args:
    point_blocks(List): center points found in each mapper.
    t1(float): distance threshold between center point and the points within a canopy. 
    t2(float): distance threshold between center point and the points within a canopy.
    cf(int): the minimum canopy size.
  '''
  canopies = []
  for block_list in point_blocks:
    for point_block in block_list:
      for i in range(point_block.shape[0]):
        point = point_block[i]
        point_strongly_bound = False
        
        for c in canopies:
          dist = np.square(c.get_center() - point).sum()
          if dist < t1: c.observe(point)
          point_strongly_bound |= (dist < t2)
          
        if not point_strongly_bound:
          canopies.append(Canopy(point))
  
  centers = []
  for c in canopies:
    c.compute_parameters()
    if c.get_num_observations() > cf:
      centers.append(c.get_center())
  
  return np.array(centers)
    
def canopy_cluster(points, t1=0.1, t2=0.1, cf=1):
  '''
  A simple implementation of canopy clustering method.
  
  Args:
    points(Expr or DistArray): the input data points matrix.
    t1(float): distance threshold between center point and the points within a canopy. 
    t2(float): distance threshold between center point and the points within a canopy.
    cf(int): the minimum canopy size.
  '''
  new_points = expr.tile_operation(points, _canopy_mapper, kw={'t1': t1, 't2': t2, 'cf': cf}).force()
  centers = find_centers(new_points.values(), t1, t2, cf)
  labels = expr.shuffle(points, _cluster_mapper, kw={'centers': centers})
  
  return labels
