#!/usr/bin/env python

'''Implementation of the ``assign`` operation.'''

import numpy as np

from .operator.region_map import region_map
from ..array import extent


def _assign_mapper(tile, ex, assign_region, value):
  '''Helper function for assign.'''
  if np.isscalar(value):
    return value

  intersection = extent.intersection(assign_region, ex)
  value_slice = extent.offset_slice(assign_region, intersection)
  region_shape = assign_region.shape
  if len(region_shape) != len(value.shape):
    j = -1
    s = []
    for axis_shape in value.shape:
      j = region_shape.index(axis_shape, j + 1)
      s.append(value_slice[j])
    value_slice = tuple(s)

  if isinstance(value, np.ndarray):
    return value[value_slice]

  return value.fetch(extent.from_slice(value_slice, value.shape))


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

  return region_map(a, region, _assign_mapper, {'assign_region': region,
                                                'value': value})
