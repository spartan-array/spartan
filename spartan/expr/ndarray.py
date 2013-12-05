from .base import Expr
from ..node import Node
from spartan.array import tile, distarray
import numpy as np

class NdArrayExpr(Expr):
  __metaclass__ = Node
  _members = ['_shape', 'dtype', 'tile_hint', 'combine_fn', 'reduce_fn']

  def __str__(self):
    return 'array(%s, %s)' % (self.shape, self.dtype)
  
  def visit(self, visitor):
    return NdArrayExpr(_shape=visitor.visit(self.shape),
                       dtype=visitor.visit(self.dtype),
                       tile_hint=self.tile_hint,
                       combine_fn=self.combine_fn,
                       reduce_fn=self.reduce_fn)
  
  def dependencies(self):
    return {}
  
  def compute_shape(self):
    return self._shape
 
  def evaluate(self, ctx, deps):
    shape = self._shape
    dtype = self.dtype
    tile_hint = self.tile_hint
    
    return distarray.create(shape, dtype,
                            combiner=self.combine_fn,
                            reducer=self.reduce_fn,
                            tile_hint=tile_hint)


def ndarray(shape, 
            dtype=np.float, 
            tile_hint=None,
            combine_fn=None, 
            reduce_fn=None):
  '''
  Lazily create a new distributed array.
  :param shape:
  :param dtype:
  :param tile_hint:
  '''
  return NdArrayExpr(_shape = shape,
                     dtype = dtype,
                     tile_hint = tile_hint,
                     combine_fn = combine_fn,
                     reduce_fn = reduce_fn) 
