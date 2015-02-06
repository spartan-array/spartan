
'''
Basic numpy style operations that are categorized to ndarray methods.

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


@disable_parakeet
def _astype_mapper(t, dtype):
  return t.astype(dtype)


def astype(x, dtype):
  '''
  Convert ``x`` to a new dtype.

  See `numpy.ndarray.astype`.

  :param x: `Expr` or `DistArray`
  :param dtype:

  '''
  assert x is not None
  return map(x, _astype_mapper, fn_kw={'dtype': np.dtype(dtype).str})


@disable_parakeet
def _tocoo(data):
  return data.tocoo()


def tocoo(array):
  '''
  Convert ``array`` to use COO (coordinate) format for tiles.

  :param array: Sparse `Expr`.
  :rtype: A new array in COO format.
  '''
  return map(array, fn=_tocoo)


def size(x, axis=None):
  '''
  Return the size (product of the size of all axes) of ``x``.

  See `numpy.ndarray.size`.

  :param x: `Expr` to compute the size of.
  '''
  if axis is None:
    return np.prod(x.shape)
  return x.shape[axis]
