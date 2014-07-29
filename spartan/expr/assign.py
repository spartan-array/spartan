#!/usr/bin/env python

'''Implementation of the ``assign`` operation.'''

import numpy as np

from .. import util
from ..array import extent
from .region_map import region_map


def _assign_mapper(tile, array, ex, value):
  '''Helper function for assign.'''
  return value


def assign(a, idx, value):
  '''Assigns ``value`` to a[index].

  :param a: DistArray
  :param idx: int, slice
  :param value: scalar, DistArray, array_like
  :rtype: ``DistArray``

  '''
  # Translate index into TileExtent.
  if np.isscalar(idx):
    idx = slice(idx, idx + 1)
  region = extent.from_slice(idx, a.shape)

  # Ensure value is an array
  if np.isscalar(value):
    a_value = np.ndarray(region.shape)
    a_value[:] = value
    value = a_value

  return region_map(a, region, _assign_mapper, {'value': value})

