#!/usr/bin/env python
'''Implementation of the higher level operator Scan.

Scan operations return ``MapExpr``s, which can be fused together. However,
because Parakeet does not currently support ``np.reshape()``, scan operations
will not compile with Parakeet.

'''
import numpy as np
import scipy.sparse as sp

from .map_with_location import map_with_location
from .optimize import disable_parakeet
from .shuffle import shuffle
from ...array import extent
from ...util import divup


def ex_to_slice(tup):
  '''Converts an extent (represented as a tuple) to a slice.'''
  return tuple([slice(ul, lr) for ul, lr in zip(tup[0], tup[1])])


def _scan_reduce_mapper(array, ex, reduce_fn, axis):
  local_reduction = reduce_fn(array.fetch(ex), axis=axis)
  axis_shape = array.tile_shape()[axis]
  id = (ex.lr[axis]-1) / axis_shape
  new_ul = list(ex.ul)
  new_lr = list(ex.lr)
  new_shape = list(ex.array_shape)
  new_ul[axis] = id
  new_lr[axis] = id + 1
  new_shape[axis] = divup(array.shape[axis], axis_shape)

  dst_ex = extent.create(new_ul, new_lr, new_shape)

  local_reduction = np.asarray(local_reduction).reshape(dst_ex.shape)
  yield (dst_ex, local_reduction)


@disable_parakeet
def _scan_mapper(tile, ex, scan_fn=None, axis=None, scan_base=None, tile_shape=None):
  '''Cannot compile with Parakeet because of ``np.reshape``.'''
  if sp.issparse(tile):
    tile = tile.todense()
  else:
    tile = tile.copy()

  base_slice = list(ex_to_slice(ex))
  new_slice = [slice(0, length) for length in tile.shape]
  if axis is None:
    axis = 1
    tile_id = (ex[1][axis] - 1) / tile_shape[axis]
    base_slice[axis] = slice(tile_id, tile_id+1)
    new_slice[axis] = slice(0, 1)
    tile[new_slice] += scan_base[base_slice]
  else:
    tile_id = (ex[1][axis] - 1) / tile_shape[axis]
    if tile_id > 0:
      base_slice[axis] = slice(tile_id-1, tile_id)
      new_slice[axis] = slice(0, 1)
      tile[new_slice] += scan_base[base_slice]

  return np.asarray(scan_fn(tile, axis=axis)).reshape(tile.shape)


def scan(array, reduce_fn=np.sum, scan_fn=np.cumsum, axis=None):
  '''
  Scan ``array`` over ``axis``.

  :param array: Expr
    The array to scan.
  :param reduce_fn: function, optional
    Local reduce function; defaults to ``np.sum``.
  :param scan_fn: function, optional
    The scan function; defaults to ``np.cumsum``.
  :param axis: int, optional
    The axis to scan; default is flattened matrix.

  :rtype: MapExpr

  '''
  reduce_result = shuffle(array, fn=_scan_reduce_mapper,
                          kw={'axis': axis if axis is not None else 1,
                              'reduce_fn': reduce_fn}, shape_hint=array.shape)
  fetch_result = reduce_result.optimized().glom()
  if axis is None:
    fetch_result = np.concatenate((np.zeros(1), scan_fn(fetch_result, axis=None)[:-1])).reshape(fetch_result.shape)
  else:
    fetch_result = scan_fn(fetch_result, axis=axis)

  return map_with_location(array, _scan_mapper,
                           fn_kw={'scan_fn': scan_fn,
                                  'axis': axis,
                                  'scan_base': fetch_result,
                                  'tile_shape': array.evaluate().tile_shape()})
