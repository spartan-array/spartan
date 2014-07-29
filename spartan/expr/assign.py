#!/usr/bin/env python

'''Implementation of the ``assign`` operation.'''

import numpy as np

from ..array import extent
from .region_map import region_map


def _assign_mapper(tile, array, ex, value):
  '''Helper function for assign.'''
  return tile


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

