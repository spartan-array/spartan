from .base import Expr
from ..node import Node, node_type
from spartan.array import tile, distarray
import numpy as np
from .. import util
from ..rpc import TimeoutException

@node_type
class NdArrayExpr(Expr):
  _members = ['_shape', 'sparse', 'dtype', 'tile_hint', 'reduce_fn']

  def __str__(self):
    return 'dist_array(%s, %s)' % (self.shape, self.dtype)
  
  def visit(self, visitor):
    return NdArrayExpr(_shape=visitor.visit(self.shape),
                       dtype=visitor.visit(self.dtype),
                       tile_hint=self.tile_hint,
                       sparse=self.sparse,
                       reduce_fn=self.reduce_fn)
  
  def dependencies(self):
    return {}
  
  def compute_shape(self):
    return self._shape
 
  def _evaluate(self, ctx, deps):
    shape = self._shape
    dtype = self.dtype
    tile_hint = self.tile_hint
    
    result = None
    try:
      result = distarray.create(shape, dtype,
                            reducer=self.reduce_fn,
                            tile_hint=tile_hint,
                            sparse=self.sparse)
    except TimeoutException as ex:
      util.log_info('ndarray expr %d need to retry' % self.expr_id)
      return self.evaluate()

    return result

def ndarray(shape, 
            dtype=np.float, 
            tile_hint=None,
            reduce_fn=None,
            sparse=False):
  '''
  Lazily create a new distributed array.
  :param shape:
  :param dtype:
  :param tile_hint:
  '''
  return NdArrayExpr(_shape = shape,
                     dtype = dtype,
                     tile_hint = tile_hint,
                     reduce_fn = reduce_fn,
                     sparse = sparse)