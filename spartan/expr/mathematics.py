'''
Basic numpy style mathematics operations on arrays.

These include --

*
'''
import sys
import __builtin__
import numpy as np
import scipy.sparse as sp

from .operator.map import map, map2
from .operator.map_with_location import map_with_location
from .operator.reduce import reduce
from .operator.ndarray import ndarray
from .operator.optimize import disable_parakeet, not_idempotent
from .. import util, blob_ctx
from ..array import extent
from ..array.extent import index_for_reduction, shapes_match
from ..util import Assert


def multiply(a, b):
  assert a.shape == b.shape
  return map((a, b), fn=lambda a, b: a.multiply(b) if sp.issparse(a) else a * b)


def power(a, b):
  return map((a, b), fn=np.power)


def add(a, b):
  return map((a, b), fn=np.add)


def sub(a, b):
  return map((a, b), fn=np.subtract)


def maximum(a, b):
  return map((a, b), np.maximum)


def ln(v):
  return map(v, fn=np.log)


def log(v):
  return map(v, fn=np.log)


def exp(v):
  return map(v, fn=np.exp)


def square(v):
  return map(v, fn=np.square)


def sqrt(v):
  return map(v, fn=np.sqrt)


def abs(v):
  return map(v, fn=np.abs)


def _sum_local(ex, data, axis):
  return data.sum(axis)


def sum(x, axis=None, tile_hint=None):
  '''
  Sum ``x`` over ``axis``.


  :param x: The array to sum.
  :param axis: Either an integer or ``None``.
  '''
  return reduce(x,
                axis=axis,
                dtype_fn=lambda input: input.dtype,
                local_reduce_fn=_sum_local,
                accumulate_fn=np.add,
                tile_hint=tile_hint)


def _prod_local(ex, data, axis):
  return data.prod(axis)


def prod(x, axis=None, tile_hint=None):
  '''
  Prod ``x`` over ``axis``.


  :param x: The array to product.
  :param axis: Either an integer or ``None``.
  '''
  return reduce(x,
                axis=axis,
                dtype_fn=lambda input: input.dtype,
                local_reduce_fn=_prod_local,
                accumulate_fn=np.multiply,
                tile_hint=tile_hint)
