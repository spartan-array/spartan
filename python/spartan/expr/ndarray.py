from .base import Expr
from .node import Node
from spartan.dense import tile, distarray
import numpy as np

class NdArrayExpr(Expr, Node):
  _members = ['_shape', 'dtype', 'tile_hint', 'combine_fn', 'reduce_fn']
  
  def dependencies(self):
    return {}
  
  def compute_shape(self):
    return self._shape
 
  def evaluate(self, ctx, prim, deps):
    shape = prim._shape
    dtype = prim.dtype
    tile_hint = prim.tile_hint
    
    if prim.combine_fn is not None:
      combiner = tile.TileAccum(prim.combine_fn)
    else:
      combiner = None
      
    if prim.reduce_fn is not None:
      reducer = tile.TileAccum(prim.reduce_fn)
    else:
      reducer = None
       
    return distarray.create(ctx, shape, dtype,
                            combiner=combiner,
                            reducer=reducer,
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
