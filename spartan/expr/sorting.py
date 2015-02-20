'''
Basic numpy style sorting, searching and counting on arrays.

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
from .statistics import max, min
from .. import util, blob_ctx
from ..array import extent
from ..array.extent import index_for_reduction, shapes_match
from ..util import Assert


def _to_structured_array(*vals):
  '''Create a structured array from the given input arrays.

  :param vals: A list of (field_name, `np.ndarray`)
  :rtype: A structured array with fields from ``kw``.
  '''
  out = np.ndarray(vals[0][1].shape,
                   dtype=','.join([a.dtype.str for name, a in vals]))
  out.dtype.names = [name for name, a in vals]
  for k, v in vals:
    out[k] = v
  return out


@disable_parakeet
def _take_idx_mapper(input):
  return input['idx']


def _dual_reducer(ex, tile, axis, idx_f=None, val_f=None):
  Assert.isinstance(ex, extent.TileExtent)
  local_idx = idx_f(tile[:], axis)
  local_val = val_f(tile[:], axis)

  global_idx = ex.to_global(local_idx, axis)
  new_idx = index_for_reduction(ex, axis)
  new_val = _to_structured_array(('idx', global_idx), ('val', local_val))

  assert shapes_match(new_idx, new_val), (new_idx, new_val.shape)
  return new_val


def _dual_combiner(a, b, op):
  return np.where(op(a['val'], b['val']), a, b)


def _dual_dtype(input):
  dtype = np.dtype('i8,%s' % np.dtype(input.dtype).str)
  dtype.names = ('idx', 'val')
  return dtype


def _arg_mapper(a, b, ex, axis=None):
  c = np.zeros(a.shape)
  c[a == b] = 1
  max_index = np.argmax(c, axis)
  if axis is not None:
    shape = list(a.shape)
    shape[axis] = 1
    global_index = max_index.reshape(tuple(shape)) + ex[0][axis]
  else:
    ex_shape = []
    for i in range(len(ex[0])):
      ex_shape.append(ex[1][i] - ex[0][i])
      ex_shape[i] = 1 if ex_shape[i] == 0 else ex_shape[i]
    local_index = extent.unravelled_pos(max_index, ex_shape)
    global_index = extent.ravelled_pos(np.asarray(ex[0]) + local_index, ex_shape)

  c = np.zeros(a.shape, dtype=np.int64) + global_index
  c[a != b] = np.prod(np.asarray(ex[2]))
  return c


def argmin(x, axis=None):
  '''
  Compute argmin over ``axis``.

  See `numpy.ndarray.argmin`.

  :param x: `Expr` to compute a minimum over.
  :param axis: Axis (integer or None).
  '''
  compute_min = min(x, axis)
  if axis is not None:
    shape = list(x.shape)
    shape[axis] = 1
    compute_min = compute_min.reshape(tuple(shape))
  argument = map_with_location((x, compute_min), _arg_mapper,
                               fn_kw={'axis': axis})
  return min(argument, axis)


def argmax(x, axis=None):
  '''
  Compute argmax over ``axis``.

  See `numpy.ndarray.argmax`.

  :param x: `Expr` to compute a maximum over.
  :param axis: Axis (integer or None).
  '''
  compute_max = max(x, axis)
  if axis is not None:
    shape = list(x.shape)
    shape[axis] = 1
    compute_max = compute_max.reshape(tuple(shape))
  argument = map_with_location((x, compute_max), _arg_mapper,
                               fn_kw={'axis': axis})
  return min(argument, axis)


def _countnonzero_local(ex, data, axis):
  if axis is None:
    if sp.issparse(data):
      return np.asarray(data.nnz)
    else:
      return np.asarray(np.count_nonzero(data))

  return (data > 0).sum(axis)


def count_nonzero(array, axis=None, tile_hint=None):
  '''
  Return the number of nonzero values in the axis of the ``array``.

  :param array: DistArray or `Expr`.
  :param axis: the axis to count
  :param tile_hint:
  :rtype: np.int64

  '''
  return reduce(array, axis,
                dtype_fn=lambda input: np.int64,
                local_reduce_fn=_countnonzero_local,
                accumulate_fn=np.add,
                tile_hint=tile_hint)


def _countzero_local(ex, data, axis):
  if axis is None:
    return np.asarray(np.prod(ex.shape) - np.count_nonzero(data))

  return (data == 0).sum(axis)


def count_zero(array, axis=None):
  '''
  Return the number of zero values in the axis of the ``array``.

  :param array: DistArray or `Expr`.
  :param axis: the axis to count
  :rtype: np.int64

  '''
  return reduce(array, axis,
                dtype_fn=lambda input: np.int64,
                local_reduce_fn=_countzero_local,
                accumulate_fn=np.add)
