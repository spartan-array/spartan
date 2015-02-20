'''
Reshape operation and expr.
'''

import numpy as np

from .operator.shuffle import shuffle
from .operator.ndarray import ndarray


def _retile_mapper(array, ex, orig_array):
  yield ex, orig_array.fetch(ex)


def retile(array, tile_hint):
  '''
  Change the tiling of ``array``, while retaining the same shape.

  Args:
    array(Expr): Array to reshape
    tile_hint(tuple): New tile shape
  '''
  tiling_type = int(tile_hint[0] == array.shape[0])
  new_array = shuffle(ndarray(array.shape, tile_hint=tile_hint).evaluate(),
                      _retile_mapper, kw={'orig_array': array}, shape_hint=array.shape,
                      cost_hint={hash(array): {'%d%d' % (tiling_type, tiling_type): 0,
                                               '%d%d' % (1-tiling_type, tiling_type): np.prod(array.shape)}})
  return new_array.optimized()
