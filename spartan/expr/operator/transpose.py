'''
Transpose operation and expr.
'''

import numpy as np
import scipy.sparse as sp

from traits.api import Instance, PythonValue

from spartan import rpc
from .base import Expr, lazify
from .shuffle import target_mapper
from ... import blob_ctx, util
from ...array import extent, tile, distarray
from ...core import LocalKernelResult
from ...util import is_iterable, Assert


def _tile_mapper(ex, **kw):
  user_fn = kw['_fn']
  fn_kw = kw['_fn_kw']
  base = kw['_base']
  base_ex = extent.create(ex.ul[::-1], ex.lr[::-1], base.shape)
  return user_fn(base_ex, **fn_kw)


class Transpose(distarray.DistArray):
  '''Transpose the underlying array base.

  Transpose does not create a copy of the base array. Instead the fetch method
  is overridden: the dimensions for the requested extent are reversed and the
  "transposed" request is sent to the underlying base array.

  Transpose supports tile_shape() by returing reversed base.tile_shape(). To
  support foreach_tile(), Transpose reports base's tiles and reverses extents'
  shape in _tile_mapper()
  '''

  def __init__(self, base):
    Assert.isinstance(base, distarray.DistArray)
    self.base = base
    self.shape = self.base.shape[::-1]
    self.dtype = base.dtype
    self.sparse = self.base.sparse
    self.tiles = base.tiles
    self.bad_tiles = []

  def tile_shape(self):
    return self.base.tile_shape()[::-1]

  def view_extent(self, ex):
    return extent.create(ex.ul[::-1], ex.lr[::-1], self.shape)

  def foreach_tile(self, mapper_fn, kw=None):
    return self.base.foreach_tile(mapper_fn=_tile_mapper,
                                  kw={'_fn_kw': kw,
                                      '_base': self,
                                      '_fn': mapper_fn})

  def extent_for_blob(self, id):
    base_ex = self.base.blob_to_ex[id]
    return extent.create(base_ex.ul[::-1], base_ex.lr[::-1], self.base.shape)

  def fetch(self, ex):
    base_ex = extent.create(ex.ul[::-1], ex.lr[::-1], self.base.shape)
    base_tile = self.base.fetch(base_ex)
    return base_tile.transpose()


class TransposeExpr(Expr):
  array = Instance(Expr)
  tile_hint = PythonValue(None, desc="Tuple or None")

  def __str__(self):
    return 'Transpose[%d] %s' % (self.expr_id, self.array)

  def _evaluate(self, ctx, deps):
    v = deps['array']

    return Transpose(v)

  def compute_shape(self):
    # May raise NotShapeable
    return self.array.shape[::-1]


def transpose(array, tile_hint=None):
  '''
  Transpose ``array``.

  Args:
    array: `Expr` to transpose.

  Returns:
    `TransposeExpr`: Transpose array.
  '''

  array = lazify(array)

  return TransposeExpr(array=array, tile_hint=tile_hint)
