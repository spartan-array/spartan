'''Primitives that backends must support.'''

from . import distarray, extent
import collections
import numpy as np

class NotShapeable(Exception):
  pass

class Primitive(object):
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
  def __init__(self, value):
    self.value = value
    
  def _shape(self):
    if isinstance(self.value, np.ndarray) or \
       isinstance(self.value, distarray.DistArray):
      return self.value.shape
    
    # Promote scalars to 0-d array
    return tuple()
    

class Map(Primitive):
  def __init__(self, inputs, map_fn):
    self.inputs = inputs
    self.map_fn = map_fn
    
  def _shape(self):
    '''Maps retain the shape of inputs.
    
    Broadcasting results in a map taking the shape of the largest input.
    '''
    shapes = [i._shape() for i in self.inputs]
    output_shape = collections.defaultdict(int)
    for s in shapes:
      for i, v in enumerate(s):
        output_shape[i] = max(output_shape[i], v)
    return tuple([output_shape[i] for i in len(output_shape)])
    
    
class Slice(Primitive):
  def __init__(self, input, idx):
    self.input = input
    self.idx = idx
  
  def _shape(self):
    return extent.shape_for_slice(self.input._shape(), self.slc)  


class Index(Primitive):
  def __init__(self, input, idx):
    self.input = input
    self.idx = idx
  
  def _shape(self):
    raise NotShapeable


class Reduce(Primitive):
  def __init__(self, input, axis, dtype_fn, local_reducer_fn, combiner_fn):
    self.input = input
    self.axis = axis
    self.dtype_fn = dtype_fn
    self.local_reducer_fn = local_reducer_fn
    self.combiner_fn = combiner_fn
    
  def _shape(self):
    return extent.shape_for_reduction(self.input._shape(), self.axis)

  