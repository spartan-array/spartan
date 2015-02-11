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


def add(a, b):
  return map((a, b), fn=np.add)


def reciprocal(a):
  return map(a, fn=np.reciprocal)


def negative(a):
  return map(a, fn=np.negative)


def sub(a, b):
  return map((a, b), fn=np.subtract)


def _rsub(a, b):
  return map((b, a), fn=np.sub)


def _multiply(a, b):
  if sp.issparse(a):
    return a.multiply(b)
  else:
    return np.multiply(a, b)


def multiply(a, b):
  return map((a, b), fn=_multiply)


def _divide(a, b):
  if sp.issparse(a):
    return a.divide(b)
  else:
    return np.divide(a, b)


def divide(a, b):
  return map((a, b), fn=_divide)


def _rdivide(a, b):
  return divide(b, a)


def true_divide(a, b):
  return map((a, b), fn=np.true_divide)


def floor_divide(a, b):
  return map((a, b), fn=np.floor_divide)


def fmod(a, b):
  return map((a, b), fn=np.fmod)


def mod(a, b):
  return map((a, b), fn=np.mod)


def remainder(a, b):
  return remainder((a, b), fn=np.remainder)


def power(a, b):
  return map((a, b), fn=np.power)


def maximum(a, b):
  return map((a, b), np.maximum)


def minimum(a, b):
  return map((a, b), np.minimum)


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


def _prod_dtype_fn(input):
  if input.dtype == np.int32:
    return np.dtype(np.int64)
  else:
    return input.dtype


def prod(x, axis=None, tile_hint=None):
  '''
  Prod ``x`` over ``axis``.


  :param x: The array to product.
  :param axis: Either an integer or ``None``.
  '''
  return reduce(x,
                axis=axis,
                dtype_fn=_prod_dtype_fn,
                local_reduce_fn=_prod_local,
                accumulate_fn=np.multiply,
                tile_hint=tile_hint)
