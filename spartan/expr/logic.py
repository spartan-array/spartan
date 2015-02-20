
'''
Basic numpy style logic operations on arrays.

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


def _all_reducer(ex, tile, axis=None):
  return np.all(tile, axis=axis)


def all(array, axis=None):
  return reduce(array,
                axis=axis,
                dtype_fn=lambda input: np.bool,
                local_reduce_fn=_all_reducer,
                accumulate_fn=np.logical_and)


def _any_reducer(ex, tile, axis=None):
  return np.any(tile, axis=axis)


def any(array, axis=None):
  return reduce(array,
                axis=axis,
                dtype_fn=lambda input: np.bool,
                local_reduce_fn=_any_reducer,
                accumulate_fn=np.logical_or)


def equal(a, b):
  return map((a, b), fn=np.equal)


def not_equal(a, b):
  return map((a, b), fn=np.not_equal)


def greater(a, b):
  return map((a, b), fn=np.greater)


def greater_equal(a, b):
  return map((a, b), fn=np.greater_equal)


def less(a, b):
  return map((a, b), fn=np.less)


def less_equal(a, b):
  return map((a, b), fn=np.less_equal)


def logical_and(a, b):
  return map((a, b), fn=np.logical_and)


def logical_or(a, b):
  return map((a, b), fn=np.logical_or)


def logical_xor(a, b):
  return map((a, b), fn=np.logical_xor)
