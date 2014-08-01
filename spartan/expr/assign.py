#!/usr/bin/env python

'''Implementation of the ``assign`` operation.'''

import numpy as np

from ..array import extent
from .region_map import region_map


def _assign_mapper(tile, array, ex, value):
  '''Helper function for assign.'''
  if np.isscalar(value):
    return value

  from ..util import log_info
  from .base import Expr

  if len(ex.shape) == 1:
    region = slice(ex.ul[0], ex.lr[0])
    return value[ex.to_slice]

  if len(value.shape) == 1:
    if ex.shape[1] > value.shape[0]:
      return value
    region = slice(ex.ul[1], ex.lr[1])
    return value[region]

  if ex.shape[1] > value.shape[1]:
    return value

  region = slice(ex.ul[1], ex.lr[1])
  return value[0][region]


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

  return region_map(a, region, _assign_mapper, {'value': value})

