'''
Transpose operation and expr.
'''

import numpy as np
import scipy.sparse as sp
from spartan import rpc
from .base import Expr, lazify
from .. import blob_ctx, util
from ..node import Node, node_type
from ..util import is_iterable, Assert
from ..array import extent, tile, distarray
from ..core import LocalKernelResult
from .shuffle import target_mapper

class Transpose(distarray.DistArray):
  '''Transpose the underlying array base.

  Transpose does not create a copy of the base array. Instead the fetch 
  method is overridden: the dimensions for the requested extent are 
  reversed and the "transposed" request is sent to the underlying base array.
  '''

  def __init__(self, base):
    Assert.isinstance(base, distarray.DistArray)
    self.base = base
    self.shape = self.base.shape[::-1]
    self.dtype = base.dtype
    self.sparse = self.base.sparse
    self.bad_tiles = []

  def fetch(self, ex):
    base_ex = extent.create(ex.ul[::-1], ex.lr[::-1], self.base.shape)
    tile = self.base.fetch(base_ex)
    return tile.transpose()

@node_type
class TransposeExpr(Expr):
  _members = ['array', 'tile_hint']

  def __str__(self):
    return 'Transpose[%d] %s' % (self.expr_id, self.expr)

  def _evaluate(self, ctx, deps):
    v = deps['array']
    shape = v.shape[::-1]

    return Transpose(v)

def transpose(array, tile_hint = None):
  '''
  Transpose ``array``.
  
  Args:
    array: `Expr` to transpose.
    
  Returns:
    `TransposeExpr`: Transpose array.
  '''

  array = lazify(array)

  return TransposeExpr(array = array, tile_hint = tile_hint)

