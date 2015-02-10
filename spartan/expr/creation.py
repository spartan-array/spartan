'''
Basic numpy style creation operations on arrays.

These include --

*
'''
import sys
import __builtin__
import numpy as np
import scipy.sparse as sp

from .operator.base import Expr
from .operator.map import map, map2
from .operator.map_with_location import map_with_location
from .operator.reduce import reduce
from .operator.ndarray import ndarray
from .operator.optimize import disable_parakeet, not_idempotent
from .. import util, blob_ctx
from ..array import extent
from ..array.extent import index_for_reduction, shapes_match
from ..util import Assert


def sparse_empty(shape, dtype=np.float32, tile_hint=None):
  '''Return an empty sparse array of the given shape.

  :param shape: `tuple`.  Shape of the resulting array.
  :param dtype: `np.dtype`
  :param tile_hint: A tuple indicating the desired tile shape for this array.
  '''
  return ndarray(shape, dtype=dtype, tile_hint=tile_hint, sparse=True)


def empty(shape, dtype=np.float32, tile_hint=None):
  '''Return an empty dense array of the given shape.

  :param shape: `tuple`.  Shape of the resulting array.
  :param dtype: `np.dtype`
  :param tile_hint: A tuple indicating the desired tile shape for this array.
  '''
  return ndarray(shape, dtype=dtype, tile_hint=tile_hint, sparse=True)


def empty_like(array, dtype=None, tile_hint=None):
  if dtype is None:
    dtype = array.dtype
  return ndarray(array.shape, dtype=dtype, tile_hint=tile_hint, sparse=array.sparse)


def _eye_mapper(tile, ex, k=None, dtype=None):
  return np.eye(ex[1][0] - ex[0][0], M=(ex[1][1] - ex[0][1]),
                k=(ex[0][0] + k), dtype=dtype)


def eye(N, M=None, k=0, dtype=np.float32, tile_hint=None):
  if M is None:
    M = N
  return map_with_location(ndarray((N, M), dtype, tile_hint),
                           _eye_mapper, fn_kw={'k': k, 'dtype': dtype})


def identity(n, dtype=np.float32, tile_hint=None):
  return eye(n, dtype=dtype, tile_hint=tile_hint)


def _make_zeros(input):
  return np.zeros(input.shape, input.dtype)


def zeros(shape, dtype=np.float32, tile_hint=None):
  '''
  Create a distributed array over the given shape and dtype, filled with zeros.

  :param shape:
  :param dtype:
  :param tile_hint:
  :rtype: `Expr`
  '''
  return map(ndarray(shape, dtype=dtype, tile_hint=tile_hint),
             fn=_make_zeros)


def zeros_like(array, dtype=None, tile_hint=None):
  if dtype is None:
    dtype = array.dtype
  if tile_hint is None:
    tile_hint = array.tile_hint if isinstance(array, Expr) else array.tile_shape()
  return zeros(array.shape, dtype=dtype, tile_hint=tile_hint)


def _make_ones(input):
  return np.ones(input.shape, input.dtype)


def ones(shape, dtype=np.float32, tile_hint=None):
  '''
  Create a distributed array over the given shape and dtype, filled with ones.

  :param shape:
  :param dtype:
  :param tile_hint:
  :rtype: `Expr`
  '''
  return map(ndarray(shape, dtype=dtype, tile_hint=tile_hint),
             fn=_make_ones)


def ones_like(array, dtype=None, tile_hint=None):
  if dtype is None:
    dtype = array.dtype
  if tile_hint is None:
    tile_hint = array.tile_hint if isinstance(array, Expr) else array.tile_shape()
  return ones(array.shape, dtype=dtype, tile_hint=tile_hint)


def _full_mapper(tile, fill_value, dtype=None):
  return np.full(tile.shape, fill_value, dtype=dtype)


def full(shape, fill_value, dtype=np.float32, tile_hint=None):
  return map(ndarray(shape, dtype=dtype, tile_hint=tile_hint),
             fn=_full_mapper)


def full_like(array, fill_value, dtype=None, tile_hint=None):
  if dtype is None:
    dtype = array.dtype
  if tile_hint is None:
    tile_hint = array.tile_hint if isinstance(array, Expr) else array.tile_shape()
  return full(array.shape, fill_value, dtype, tile_hint)


@disable_parakeet
def _arange_mapper(tile, ex, start, stop, step, dtype=None):
  pos = extent.ravelled_pos(ex[0], ex[2])
  ex_start = pos*step + start
  ex_stop = np.prod(tile.shape)*step + ex_start

  # np.reshape is not supported by parakeet.
  return np.arange(ex_start, ex_stop, step, dtype=dtype).reshape(tile.shape)


def arange(start=None, stop=None, step=1, dtype=np.float, tile_hint=None):
  '''
  An extended version of `np.arange`.

  Returns a new array of the given shape and dtype. Values of the
  array are equivalent to running: ``np.arange(np.prod(shape)).reshape(shape)``.

  Shape xor stop must be supplied. If shape is supplied, stop is calculated
  using the shape, start, and step (if start and step are given). If stop is
  supplied, then the resulting Expr is a 1d array with length calculated via
  start, stop, and step.

  :param shape: tuple, optional
    The shape of the resulting Expr: e.x.(10, ) and (3, 5). Shape xor stop
    must be supplied.
  :param start: number, optional
    Start of interval, including this value. The default start value is 0.
  :param stop: number, optional
    End of interval, excluding this value. Shape xor stop must be supplied.
  :param step: number, optional
    Spacing between values. The default step size is 1.
  :param dtype: dtype
    The type of the output array.
  :param tile_hint:

  :rtype: `Expr`

  Examples:
  sp.arange((3, 5)) == np.arange(15).reshape((3, 5))
  sp.arange(None, stop=10) == np.arange(10)
  sp.arange((3, 5), -1) == np.arange(-1, 14).reshape((3, 5))
  sp.arange((3, 5), step=2) == np.arange(0, 30, 2).reshape((3, 5))
  '''
  if start is None and stop is None:
    raise ValueError('No valid parameters')

  shape = None
  if isinstance(start, (tuple, list)):
    shape = start
    start = 0
    if stop is not None:
      start = stop
      stop = None
  elif start is None:
    start = 0
  elif stop is None:
    stop = start
    start = 0

  if shape is None and stop is None:
    raise ValueError('Shape or stop expected, none supplied.')

  if shape is not None and stop is not None:
    raise ValueError('Only shape OR stop can be supplied, not both.')

  if shape is None:
    # Produces 1d array based on start, stop, step
    length = int(np.ceil((stop - start) / float(step)))
    shape = (length, )

  return map_with_location(ndarray(shape, dtype, tile_hint), _arange_mapper,
                           fn_kw={'start': start, 'stop': stop,
                                  'step': step, 'dtype': dtype})


def _make_sparse_diagonal(tile, ex):
  ul, lr = ex[0], ex[1]
  data = sp.lil_matrix(tile.shape, dtype=tile.dtype)

  if ul[0] >= ul[1] and ul[0] < lr[1]:  # below the diagonal
    for i in range(ul[0], __builtin__.min(lr[0], lr[1])):
      data[i - ul[0], i - ul[1]] = 1
  elif ul[1] >= ul[0] and ul[1] < lr[0]:  # above the diagonal
    for j in range(ul[1], __builtin__.min(lr[1], lr[0])):
      data[j - ul[0], j - ul[1]] = 1

  return data


def sparse_diagonal(shape, dtype=np.float32, tile_hint=None):
  return map_with_location(ndarray(shape, dtype, tile_hint, sparse=True),
                           _make_sparse_diagonal)


def _diagflat_mapper(extents, tiles, shape=None):
  '''Create a diagonal array section for this extent.

  If the extent does not lie on the diagonal, a zero array is returned.

  :param array: DistArray
  :param ex: Extent
    Region being processed.
  '''
  ex = extents[0]
  tile = tiles[0]
  head = extent.ravelled_pos(ex.ul, ex.array_shape)
  tail = extent.ravelled_pos([l - 1 for l in ex.lr], ex.array_shape)

  result = np.diagflat(tile)
  if head != 0:
    result = np.hstack((np.zeros(((tail - head + 1), head)), result))
  if tail + 1 != shape[0]:
    result = np.hstack((result, np.zeros((tail - head + 1, shape[0] - (tail + 1)))))

  target_ex = extent.create((head, 0), (tail + 1, shape[1]), shape)
  yield target_ex, result


def diagflat(array):
  '''
  Create a diagonal array with the given data on the diagonal
  the shape should be (array.shape[0] * array.shape[1]) x (array.shape[0] * array.shape[1])

  :param array: 2D DistArray
    The data to fill the diagonal.
  '''
  shape = (np.prod(array.shape), np.prod(array.shape))
  return map2(array, 0, fn=_diagflat_mapper, fn_kw={'shape': shape}, shape=shape)


def _diagonal_mapper(ex, tiles, shape=None):
  tile = tiles[0]
  max_dim = __builtin__.max(*ex.ul)
  first_point = [max_dim for i in range(len(ex.ul))]
  slices = []
  for i in range(len(ex.ul)):
    if first_point[i] >= ex.lr[i]:
      return
    slices.append(slice(first_point[i] - ex.ul[i], ex.shape[i]))

  result = tile[slices].diagonal()
  target_ex = extent.create((first_point[0], ),
                            (first_point[0] + result.shape[0], ),
                            shape)
  yield target_ex, result


def diagonal(a):
  '''Return specified diagonals.

  :param a: array_like
    Array from which the diagonals are taken.
  :rtype Map2Expr

  Raises
  ------
  ValueError
    If the dimension of `a` is less than 2.

  '''
  if len(a.shape) < 2:
    raise ValueError("diag requires an array of at least two dimensions")

  shape = (__builtin__.min(a.shape), )
  return map2(a, fn=_diagonal_mapper, fn_kw={'shape': shape}, shape=shape)


def diag(array, offset=0):
  '''
  Extract a diagonal or construct a diagonal array.

  :param array: array_like
    Array from which the diagonals are taken.
  :param offset: int, optional
    Diagonal in question. The default is 0. Use k>0 for diagonals
    above the main diagonal, and k<0 for diagonals below the main diagonal.
    This argument hasn't been implemented yet.


  :rtype Map2Expr

  Raises
  ------
  ValueError
    If the dimension of `array` is not 1 or 2.
  NotImplementedError
    If offset is being set.
  '''
  if offset != 0:
    raise NotImplementedError

  if len(array.shape) == 1:
    return diagflat(array)
  elif len(array.shape) == 2:
    return diagonal(array)
  else:
    raise ValueError("Input must be 1- or 2-d.")
