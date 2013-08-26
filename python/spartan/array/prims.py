'''Primitives that backends must support.'''

from . import distarray, extent
from spartan.node import Node
from spartan.util import Assert
import collections
import numpy as np

class NotShapeable(Exception):
  pass

class Primitive(Node):
  _members = []
  cached_value = None
  
  def shape(self):
    '''Try to compute the shape of this DAG.
    
    If the value has been computed already this always succeeds.
    '''
    if self.cached_value is not None:
      return self.cached_value.shape
    
    try:
      return self._shape()
    except NotShapeable:
      return None
    
  def typename(self):
    return self.__class__.__name__
    

class Value(Primitive):
  _members = ['value']
  
  def dependencies(self):
    return []

  def _shape(self):
    if isinstance(self.value, np.ndarray) or \
       isinstance(self.value, distarray.DistArray):
      return self.value.shape
    
    # Promote scalars to 0-d array
    return tuple()
    

class MapTiles(Primitive):
  _members = ['inputs', 'map_fn']
  
  def dependencies(self):
    return self.inputs
  
  def _shape(self):
    '''Maps retain the shape of inputs.
    
    Broadcasting results in a map taking the shape of the largest input.
    '''
    shapes = [i._shape() for i in self.deps]
    output_shape = collections.defaultdict(int)
    for s in shapes:
      for i, v in enumerate(s):
        output_shape[i] = max(output_shape[i], v)
    return tuple([output_shape[i] for i in len(output_shape)])


class MapExtents(MapTiles):
  pass    
    
class Slice(Primitive):
  _members = ['src', 'idx']
  
  def dependencies(self):
    return [self.src, self.idx]
  
  def _shape(self):
    return extent.shape_for_slice(self.input._shape(), self.slc)  


class Index(Primitive):
  _members = ['src', 'idx']
  
  def dependencies(self):
    return [self.src, self.idx]
  
  def _shape(self):
    raise NotShapeable


class Reduce(Primitive):
  _members = ['input', 'axis', 'dtype_fn', 'local_reducer_fn', 'combiner_fn']
    
  def dependencies(self):
    return [self.input]
  
  def _shape(self):
    return extent.shape_for_reduction(self.input._shape(), self.axis)

  
class NewArray(Primitive):
  _members = ['array_shape', 'dtype']
  
  def dependencies(self):
    return []
  
  def _shape(self):
    return self.array_shape