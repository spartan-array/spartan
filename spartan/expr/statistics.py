'''
Basic numpy style statistics operations on arrays.

These include --

*
'''
import sys
import __builtin__
import numpy as np
import scipy.sparse as sp
import scipy.stats

from .mathematics import sqrt, sum
from .operator.map import map, map2
from .operator.map_with_location import map_with_location
from .operator.reduce import reduce
from .operator.ndarray import ndarray
from .operator.optimize import disable_parakeet, not_idempotent
from .. import util, blob_ctx
from ..array import extent
from ..array.extent import index_for_reduction, shapes_match
from ..util import Assert


def max(x, axis=None, tile_hint=None):
  '''Compute the maximum value over an array (or axis).  See `numpy.max`.

  Arguments:
    x (Expr):
    axis (int, tuple, or None): Axis to operate over
    tile_hint (tuple or None): Tile shape for the output array.

  Returns:
   Expr:
  '''
  return reduce(x,
                axis=axis,
                dtype_fn=lambda input: input.dtype,
                local_reduce_fn=lambda ex, data, axis: data.max(axis),
                accumulate_fn=np.maximum,
                tile_hint=tile_hint)


def min(x, axis=None, tile_hint=None):
  '''Compute the minimum value over an array (or axis).  See `numpy.min`.

  Arguments:
    x (Expr):
    axis (int, tuple, or None): Axis to operate over
    tile_hint (tuple or None): Tile shape for the output array.

  Returns:
   Expr:
  '''
  return reduce(x,
                axis=axis,
                dtype_fn=lambda input: input.dtype,
                local_reduce_fn=lambda ex, data, axis: data.min(axis),
                accumulate_fn=np.minimum,
                tile_hint=tile_hint)


def mean(x, axis=None):
  '''
  Compute the mean of ``x`` over ``axis``.

  See `numpy.ndarray.mean`.

  :param x: `Expr`
  :param axis: integer or ``None``
  '''
  if axis is None:
    return sum(x, axis) / np.prod(x.shape)
  else:
    return sum(x, axis) / x.shape[axis]


def _num_tiles(array):
  '''Calculate the number of tiles for a given DistArray.'''
  num_tiles = util.divup(array.shape[0], array.tile_shape()[0])
  remaining = (array.shape[1] - array.tile_shape()[1]) * num_tiles
  return num_tiles + util.divup(remaining, array.tile_shape()[1])


def std(a, axis=None):
  '''Compute the standard deviation along the specified axis.

  Returns the standard deviation of the array elements. The standard deviation
  is computed for the flattened array by default, otherwise over the specified
  axis.

  :param a: array_like
    Calculate the standard deviation of these values.
  :axis: int, optional
    Axis along which the standard deviation is computed. The default is to
    compute the standard deviation of the flattened array.

  :rtype standard_deviation: Expr
  '''
  a_casted = a.astype(np.float64)
  return sqrt(mean(a_casted ** 2, axis) - mean(a_casted, axis) ** 2)


def _bincount_mapper(ex, tiles, minlength=None):
  if len(tiles) > 1:
    result = np.bincount(tiles[0], weights=tiles[1], minlength=minlength)
  else:
    result = np.bincount(tiles[0], minlength=minlength)
  result_ex = extent.from_shape(result.shape)
  yield result_ex, result


def bincount(v, weights=None, minlength=None):
  '''
  Count unique values in ``v``.
  See `numpy.bincount` for more information.

  Arguments:
    v (Expr): Array of non-negative integers
  Returns:
    Expr: Integer array of counts.
  '''
  minval = min(v).glom()
  maxval = max(v).glom()
  assert minval > 0
  if minlength is not None:
    minlength = __builtin__.max(maxval + 1, minlength)
  else:
    minlength = maxval + 1

  if weights is not None:
    return map2((v, weights), fn=_bincount_mapper, fn_kw={'minlength': minlength},
                shape=(minlength,), reducer=np.add)
  else:
    return map2(v, fn=_bincount_mapper, fn_kw={'minlength': minlength},
                shape=(minlength,), reducer=np.add)


def _normalize_mapper(tile, ex, axis, norm_value):
  '''Normalize a region of an array.

  Returns a new, normalized region.

  :param value: np.ndarray
    Data being processed.
  :param ex: tuple
    The value's location in the global array (ul, lr, array_shape).
  :param axis: int, optional
    The axis to normalize; defaults to flattened array.

  '''
  ul = ex[0]
  if axis is None:
    tile /= norm_value
  elif axis == 0:
    tile[:, 0] /= norm_value[ul[1]]
  elif axis == 1:
    tile[0, :] /= norm_value[ul[0]]

  return tile


def normalize(array, axis=None):
  '''Normalize the values of ``array`` over axis.

  After normalization `sum(array, axis)` will be equal to 1.

  :param array: Expr
    The array to be normalized.
  :param axis: int, optional
    The axis to normalize.``None`` will normalize the flattened array.

  :rtype: MapExpr
    Normalized array.

  '''
  axis_sum = sum(array, axis=axis).glom()
  return map_with_location(array, _normalize_mapper,
                           fn_kw={'axis': axis, 'norm_value': axis_sum})


def norm(array, ord=2):
  '''
  Norm of ``array``.

  The following norms can be calculated:
  =====  ============================  ==========================
  ord    norm for matrices             norm for vectors
  =====  ============================  ==========================
  1      max(sum(abs(array), axis=0))  sum(abs(array))
  2      not support                   sum(abs(array)**2)**(1/2)
  =====  ============================  ==========================

  Args:
    array (Expr): input array
    ord (int): ord must be in {1,2}, the order of the norm.

  Returns:
    `Expr`: Normed array.
  '''
  assert ord == 1 or ord == 2

  if ord == 1:
    result = reduce(array,
                    axis=0,
                    dtype_fn=lambda input: input.dtype,
                    local_reduce_fn=lambda ex, data, axis: np.abs(data).sum(axis),
                    accumulate_fn=np.add).glom()
    return np.max(result)
  elif len(array.shape) == 1 or len(array.shape) == 2 and array.shape[1] == 1:
    result = reduce(array,
                    axis=0,
                    dtype_fn=lambda input: input.dtype,
                    local_reduce_fn=lambda ex, data, axis: np.square(data).sum(axis),
                    accumulate_fn=np.add).glom()
    return np.sqrt(result)

  assert False, "matrix norm-2 is not support!"


def norm_cdf(v):
  return map(v, fn=scipy.stats.norm.cdf, numpy_expr='mathlib.norm_cdf')
