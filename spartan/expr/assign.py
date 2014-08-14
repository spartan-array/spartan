#!/usr/bin/env python

'''Implementation of the ``assign`` operation.'''

import numpy as np

from ..array import extent
from .region_map import region_map


def _assign_mapper(tile, ex, array, start, value):
  '''Helper function for assign.'''
  if np.isscalar(value) or isinstance(value, np.ndarray):
    return value

  value_ul = [0] * min(len(ex.shape), len(value.shape))
  value_lr = [0] * len(value_ul)
  offset = max(len(ex.shape) - len(value.shape), 0)
  for i in range(len(value_ul)):
    value_ul[i] = max(ex.ul[i + offset] - start[i + offset], 0)
    value_lr[i] = min(ex.lr[i + offset], value.shape[i])

  return value.fetch(extent.create(value_ul, value_lr, value.shape))


def assign(a, idx, value):
  '''Assigns ``value`` to a[index].

  :param a: DistArray
  :param idx: int, slice, tuple, TileExtent
  :param value: scalar, DistArray, array_like
  :rtype: ``DistArray``

  '''
  # Translate index into TileExtent.
  if not isinstance(idx, extent.TileExtent):
    if np.isscalar(idx):
      idx = slice(idx, idx + 1)
    region = extent.from_slice(idx, a.shape)

  return region_map(a, region, _assign_mapper, {'start': region.ul,
                                                'value': value})

