'''Primitives that backends must support.'''

from . import distarray, extent
from .node import node_type
from spartan.util import Assert
import collections
import numpy as np


class NotShapeable(Exception):
  pass

class Primitive(object):
  _members = []
  cached_value = None
  
  def shape(self):
    '''Try to compute the shape of this DAG.
    
    If the value has been computed already this always succeeds.
    '''
    if self.cached_value is not None:
      return self.cached_value.shape
    return self._shape()
    
  def typename(self):
    return self.__class__.__name__
    
@node_type
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
  
@node_type
class Map(Primitive):
   
  def dependencies(self):
    return self.inputs
  

@node_type
class MapTiles(Map):
  _members = ['inputs', 'map_fn', 'fn_args', 'fn_kw', ]
  
  def _shape(self):
    '''MapTiles retains the shape of inputs.
    
    Broadcasting results in a map taking the shape of the largest input.
    '''
    shapes = [i._shape() for i in self.dependencies()]
    output_shape = collections.defaultdict(int)
    for s in shapes:
      for i, v in enumerate(s):
        output_shape[i] = max(output_shape[i], v)
    return tuple([output_shape[i] for i in range(len(output_shape))])


@node_type
class MapExtents(Map):
  _members = ['inputs', 'map_fn', 'fn_args', 'fn_kw']
  def _shape(self):
    raise NotShapeable


@node_type
class Slice(Primitive):
  _members = ['src', 'idx']
  
  def dependencies(self):
    return [self.src, self.idx]
  
  def _shape(self):
    return extent.shape_for_slice(self.input._shape(), self.slc)  

@node_type
class Index(Primitive):
  _members = ['src', 'idx']
  
  def dependencies(self):
    return [self.src, self.idx]
  
  def _shape(self):
    raise NotShapeable


@node_type
class Reduce(Primitive):
  _members = ['input', 'axis', 'dtype_fn', 'local_reducer_fn', 'combiner_fn']
    
  def dependencies(self):
    return self.input
  
  def _shape(self):
    return extent.shape_for_reduction(self.input._shape(), self.axis)


@node_type
class NewArray(Primitive):
  _members = ['array_shape', 'dtype', 'tile_hint']
  
  def dependencies(self):
    return []
  
  def _shape(self):
    return self.array_shape