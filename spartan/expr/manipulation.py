'''
Basic numpy style manipulation operations on arrays.

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


def _ravel_mapper(ex, tiles):
  ul = extent.ravelled_pos(ex.ul, ex.array_shape)
  lr = 1 + extent.ravelled_pos([lr - 1 for lr in ex.lr], ex.array_shape)
  shape = (np.prod(ex.array_shape),)

  ravelled_ex = extent.create((ul,), (lr,), shape)
  ravelled_data = tiles[0].ravel()
  yield ravelled_ex, ravelled_data


def ravel(v):
  '''
  "Ravel" ``v`` to a one-dimensional array of shape (size(v),).

  See `numpy.ndarray.ravel`.
  :param v: `Expr` or `DistArray`
  '''
  return map2(v,  fn=_ravel_mapper, shape=(np.prod(v.shape),))


def _concatenate_mapper(extents, tiles, shape=None, axis=0):
  if len(extents[0].shape) > 1:
    ul = extents[0].ul
    lr = list(extents[0].lr)
    lr[axis] += extents[1].shape[axis]
    ex = extent.create(ul, lr, shape)
    yield ex, np.concatenate((tiles[0], tiles[1]), axis=axis)
  else:
    ex = extent.create(extents[0].ul, extents[0].lr, shape)
    yield ex, tiles[0]
    ul = (extents[0].array_shape[0] + extents[1].ul[0], )
    lr = (extents[0].array_shape[0] + extents[1].lr[0], )
    ex = extent.create(ul, lr, shape)
    yield ex, tiles[1]


def concatenate(a, b, axis=0):
  '''Join two arrays together.'''
  # Calculate the shape of the resulting matrix and check dimensions.
  new_shape = [0] * len(a.shape)
  for index, (dim1, dim2) in enumerate(zip(a.shape, b.shape)):
    if index == axis:
      new_shape[index] = dim1 + dim2
      continue
    new_shape[index] = dim1
    if dim1 != dim2:
      raise ValueError('all the input array dimensions except for the'
                       'concatenation axis must match exactly')

  if len(a.shape) > 1:
    partition_axis = extent.largest_dim_axis(a.shape, exclude_axes=[axis])
  else:
    partition_axis = 0

  return map2((a, b), (partition_axis, partition_axis), fn=_concatenate_mapper,
              fn_kw={'axis': axis, 'shape': new_shape}, shape=new_shape)
