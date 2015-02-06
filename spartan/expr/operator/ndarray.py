import numpy as np

from traits.api import Tuple, Bool, PythonValue

from spartan.array import distarray
from .base import Expr, expr_like


class NdArrayExpr(Expr):
  _shape = Tuple
  sparse = Bool
  dtype = PythonValue(None, desc="np.type or type")
  tile_hint = PythonValue(None, desc="Tuple or None")
  reduce_fn = PythonValue(None, desc="Function or None")

  def pretty_str(self):
    return 'DistArray[%d](%s, %s, hint=%s)' % (self.expr_id, self.shape, np.dtype(self.dtype).name, self.tile_hint)

  def visit(self, visitor):
    return expr_like(self,
      _shape=visitor.visit(self.shape),
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

    return distarray.create(shape, dtype,
                            reducer=self.reduce_fn,
                            tile_hint=tile_hint,
                            sparse=self.sparse)

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
