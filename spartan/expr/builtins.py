'''
Basic numpy style operations on arrays.

These include --

* Array creation routines: (`rand`, `randn`, `zeros`, `ones`, `arange`)
* Reductions: (`sum`, `argmin`, `argmax`, `mean`)
* Shape/type casting: (`reshape`, `ravel`, `astype`, `shape`, `size`)
* Other: (`dot`).
'''
import sys

import numpy as np
import scipy.sparse as sp

from .. import util, blob_ctx
from ..array import extent
from ..array.extent import index_for_reduction, shapes_match
from ..util import Assert
from .map import map, map2
from .map_with_location import map_with_location
from .outer import outer
from .ndarray import ndarray
from .optimize import disable_parakeet, not_idempotent
from .reduce import reduce
import __builtin__


@disable_parakeet
def _set_random_seed_mapper(input):
  import time
  import random
  import os
  np.random.seed((int(time.time() * 100000) + random.randint(0, 10000000) +
                  os.getpid()) % 4294967295)
  return np.zeros((1, ))


def set_random_seed():
  ctx = blob_ctx.get()
  map(ndarray((ctx.num_workers, ), dtype=np.int32,
              tile_hint=(1, )), fn=_set_random_seed_mapper).force()


@disable_parakeet
def _make_rand(input):
  return np.random.rand(*input.shape)


@disable_parakeet
def _make_randn(input):
  return np.random.randn(*input.shape)


@disable_parakeet
def _make_randint(input, low=0, high=10):
  return np.random.randint(low, high, size=input.shape)


@disable_parakeet
def _make_sparse_rand(input,
                      density=None,
                      dtype=None,
                      format='csr'):
  Assert.eq(len(input.shape), 2)

  return sp.rand(input.shape[0],
                 input.shape[1],
                 density=density,
                 format=format,
                 dtype=dtype)


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


@not_idempotent
def rand(*shape, **kw):
  '''
  Return a random array sampled from the uniform distribution on [0, 1).

  :param tile_hint: A tuple indicating the desired tile shape for this array.
  '''
  tile_hint = None
  if 'tile_hint' in kw:
    tile_hint = kw['tile_hint']
    del kw['tile_hint']

  assert len(kw) == 0, 'Unknown keywords %s' % kw

  for s in shape: assert isinstance(s, (int, long))
  return map(ndarray(shape, dtype=np.float, tile_hint=tile_hint),
             fn=_make_rand)


@not_idempotent
def randn(*shape, **kw):
  '''
  Return a random array sampled from the standard normal distribution.

  :param tile_hint: A tuple indicating the desired tile shape for this array.
  '''
  tile_hint = None
  if 'tile_hint' in kw:
    tile_hint = kw['tile_hint']
    del kw['tile_hint']

  for s in shape: assert isinstance(s, (int, long))
  return map(ndarray(shape, dtype=np.float, tile_hint=tile_hint), fn=_make_randn)


@not_idempotent
def randint(*shape, **kw):
  '''
  Return a random integer array from the "discrete uniform" distribution in the interval [`low`, `high`).

  :param low: Lowest (signed) integer to be drawn from the distribution.
  :param high: Largest (signed) integer to be drawn from the distribution.
  :param tile_hint: A tuple indicating the desired tile shape for this array.
  '''
  tile_hint = None
  if 'tile_hint' in kw:
    tile_hint = kw['tile_hint']
    del kw['tile_hint']

  for s in shape: assert isinstance(s, (int, long))
  return map(ndarray(shape, dtype=np.float, tile_hint=tile_hint), fn=_make_randint, fn_kw=kw)


@not_idempotent
def sparse_rand(shape,
                density=0.001,
                format='lil',
                dtype=np.float32,
                tile_hint=None):
  '''Make a distributed sparse random array.

  Random values are chosen from the uniform distribution on [0, 1).

  Args:
    density(float): Fraction of values to be filled
    format(string): Sparse tile format (lil, coo, csr, csc).
    dtype(np.dtype): Datatype of array.
    tile_hint(tuple or None): Shape of array tiles.

  Returns:
    Expr:
  '''

  for s in shape: assert isinstance(s, (int, long))
  return map(ndarray(shape, dtype=dtype, tile_hint=tile_hint, sparse=True),
             fn=_make_sparse_rand,
             fn_kw={'dtype': dtype,
                    'density': density,
                    'format': format})


def sparse_empty(shape, dtype=np.float32, tile_hint=None):
  '''Return an empty sparse array of the given shape.

  :param shape: `tuple`.  Shape of the resulting array.
  :param dtype: `np.dtype`
  :param tile_hint: A tuple indicating the desired tile shape for this array.
  '''
  return ndarray(shape, dtype=dtype, tile_hint=tile_hint, sparse=True)


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


def _normalize_mapper(tile, ex, axis, norm_value):
  '''Normalize a region of an array.

  Returns a new, normalized region.

  :param value: np.ndarray
    Data being processed.
  :param ex: tuple
    The value's location in the global array (ul, lr, array_shape).
  :param axis: int, optional
    The axis to normalize; defaults to flattened array.

  '''
  ul = ex[0]
  if axis is None:
    tile /= norm_value
  elif axis == 0:
    tile[:, 0] /= norm_value[ul[1]]
  elif axis == 1:
    tile[0, :] /= norm_value[ul[0]]

  return tile


def normalize(array, axis=None):
  '''Normalize the values of ``array`` over axis.

  After normalization `sum(array, axis)` will be equal to 1.

  :param array: Expr
    The array to be normalized.
  :param axis: int, optional
    The axis to normalize.``None`` will normalize the flattened array.

  :rtype: MapExpr
    Normalized array.

  '''
  axis_sum = sum(array, axis=axis).glom()
  return map_with_location(array, _normalize_mapper,
                           fn_kw={'axis': axis, 'norm_value': axis_sum})


def norm(array, ord=2):
  '''
  Norm of ``array``.

  The following norms can be calculated:
  =====  ============================  ==========================
  ord    norm for matrices             norm for vectors
  =====  ============================  ==========================
  1      max(sum(abs(array), axis=0))  sum(abs(array))
  2      not support                   sum(abs(array)**2)**(1/2)
  =====  ============================  ==========================

  Args:
    array (Expr): input array
    ord (int): ord must be in {1,2}, the order of the norm.

  Returns:
    `Expr`: Normed array.
  '''
  assert ord == 1 or ord == 2

  if ord == 1:
    result = reduce(array,
                    axis=0,
                    dtype_fn=lambda input: input.dtype,
                    local_reduce_fn=lambda ex, data, axis: np.abs(data).sum(axis),
                    accumulate_fn=np.add).glom()
    return np.max(result)
  elif len(array.shape) == 1 or len(array.shape) == 2 and array.shape[1] == 1:
    result = reduce(array,
                    axis=0,
                    dtype_fn=lambda input: input.dtype,
                    local_reduce_fn=lambda ex, data, axis: np.square(data).sum(axis),
                    accumulate_fn=np.add).glom()
    return np.sqrt(result)

  assert False, "matrix norm-2 is not support!"


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


def _make_ones(input):
  return np.ones(input.shape, input.dtype)


def _make_zeros(input):
  return np.zeros(input.shape, input.dtype)


def zeros(shape, dtype=np.float, tile_hint=None):
  '''
  Create a distributed array over the given shape and dtype, filled with zeros.

  :param shape:
  :param dtype:
  :param tile_hint:
  :rtype: `Expr`
  '''
  return map(ndarray(shape, dtype=dtype, tile_hint=tile_hint),
             fn=_make_zeros)


def ones(shape, dtype=np.float, tile_hint=None):
  '''
  Create a distributed array over the given shape and dtype, filled with ones.

  :param shape:
  :param dtype:
  :param tile_hint:
  :rtype: `Expr`
  '''
  return map(ndarray(shape, dtype=dtype, tile_hint=tile_hint),
             fn=_make_ones)


@disable_parakeet
def _arange_mapper(tile, ex, start, stop, step, dtype=None):
  pos = extent.ravelled_pos(ex[0], ex[2])
  ex_start = pos*step + start
  ex_stop = np.prod(tile.shape)*step + ex_start

  # np.reshape is not supported by parakeet.
  return np.arange(ex_start, ex_stop, step, dtype=dtype).reshape(tile.shape)


def arange(shape=None, start=0, stop=None, step=1, dtype=np.float, tile_hint=None):
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
  if shape is None and stop is None:
    raise ValueError('Shape or stop expected, none supplied.')

  if shape is not None and stop is not None:
    raise ValueError('Only shape OR stop can be supplied, not both.')

  if shape is None:
    # Produces 1d array based on start, stop, step
    length = int(np.ceil((stop - start) / float(step)))
    shape = (length, )

  if stop is None:
    stop = step*(np.prod(shape) + start)

  return map_with_location(ndarray(shape, dtype, tile_hint), _arange_mapper,
                           fn_kw={'start': start, 'stop': stop,
                                  'step': step, 'dtype': dtype})


def _sum_local(ex, data, axis):
  #util.log_info('Summing: %s %s', ex, axis)
  #util.log_info('Summing: %s', data.shape)
  #util.log_info('Result: %s', data.sum(axis).shape)
  return data.sum(axis)


def sum(x, axis=None, tile_hint=None):
  '''
  Sum ``x`` over ``axis``.


  :param x: The array to sum.
  :param axis: Either an integer or ``None``.
  '''
  return reduce(x,
                axis=axis,
                dtype_fn=lambda input: input.dtype,
                local_reduce_fn=_sum_local,
                accumulate_fn=np.add,
                tile_hint=tile_hint)


def max(x, axis=None, tile_hint=None):
  '''Compute the maximum value over an array (or axis).  See `numpy.max`.

  Arguments:
    x (Expr):
    axis (int, tuple, or None): Axis to operate over
    tile_hint (tuple or None): Tile shape for the output array.

  Returns:
   Expr:
  '''
  return reduce(x,
                axis=axis,
                dtype_fn=lambda input: input.dtype,
                local_reduce_fn=lambda ex, data, axis: data.max(axis),
                accumulate_fn=np.maximum,
                tile_hint=tile_hint)


def min(x, axis=None, tile_hint=None):
  '''Compute the minimum value over an array (or axis).  See `numpy.min`.

  Arguments:
    x (Expr):
    axis (int, tuple, or None): Axis to operate over
    tile_hint (tuple or None): Tile shape for the output array.

  Returns:
   Expr:
  '''
  return reduce(x,
                axis=axis,
                dtype_fn=lambda input: input.dtype,
                local_reduce_fn=lambda ex, data, axis: data.min(axis),
                accumulate_fn=np.minimum,
                tile_hint=tile_hint)


def mean(x, axis=None):
  '''
  Compute the mean of ``x`` over ``axis``.

  See `numpy.ndarray.mean`.

  :param x: `Expr`
  :param axis: integer or ``None``
  '''
  if axis is None:
    return sum(x, axis) / np.prod(x.shape)
  else:
    return sum(x, axis) / x.shape[axis]


def _num_tiles(array):
  '''Calculate the number of tiles for a given DistArray.'''
  num_tiles = util.divup(array.shape[0], array.tile_shape()[0])
  remaining = (array.shape[1] - array.tile_shape()[1]) * num_tiles
  return num_tiles + util.divup(remaining, array.tile_shape()[1])


def std(a, axis=None):
  '''Compute the standard deviation along the specified axis.

  Returns the standard deviation of the array elements. The standard deviation
  is computed for the flattened array by default, otherwise over the specified
  axis.

  :param a: array_like
    Calculate the standard deviation of these values.
  :axis: int, optional
    Axis along which the standard deviation is computed. The default is to
    compute the standard deviation of the flattened array.

  :rtype standard_deviation: Expr
  '''
  a_casted = a.astype(np.float64)
  return sqrt(mean(a_casted ** 2, axis) - mean(a_casted, axis) ** 2)  #.optimized()


def _to_structured_array(*vals):
  '''Create a structured array from the given input arrays.

  :param vals: A list of (field_name, `np.ndarray`)
  :rtype: A structured array with fields from ``kw``.
  '''
  out = np.ndarray(vals[0][1].shape,
                   dtype=','.join([a.dtype.str for name, a in vals]))
  out.dtype.names = [name for name, a in vals]
  for k, v in vals:
    out[k] = v
  return out


@disable_parakeet
def _take_idx_mapper(input):
  return input['idx']


def _dual_reducer(ex, tile, axis, idx_f=None, val_f=None):
  Assert.isinstance(ex, extent.TileExtent)
  local_idx = idx_f(tile[:], axis)
  local_val = val_f(tile[:], axis)

  global_idx = ex.to_global(local_idx, axis)
  new_idx = index_for_reduction(ex, axis)
  new_val = _to_structured_array(('idx', global_idx), ('val', local_val))

  assert shapes_match(new_idx, new_val), (new_idx, new_val.shape)
  return new_val


def _dual_combiner(a, b, op):
  return np.where(op(a['val'], b['val']), a, b)


def _dual_dtype(input):
  dtype = np.dtype('i8,%s' % np.dtype(input.dtype).str)
  dtype.names = ('idx', 'val')
  return dtype


def _arg_mapper(a, b, ex, axis=None):
  c = np.zeros(a.shape)
  c[a == b] = 1
  max_index = np.argmax(c, axis)
  if axis is not None:
    shape = list(a.shape)
    shape[axis] = 1
    global_index = max_index.reshape(tuple(shape)) + ex[0][axis]
  else:
    ex_shape = []
    for i in range(len(ex[0])):
      ex_shape.append(ex[1][i] - ex[0][i])
      ex_shape[i] = 1 if ex_shape[i] == 0 else ex_shape[i]
    local_index = extent.unravelled_pos(max_index, ex_shape)
    global_index = extent.ravelled_pos(np.asarray(ex[0]) + local_index, ex_shape)

  c = np.zeros(a.shape, dtype=np.int64) + global_index
  c[a != b] = np.prod(np.asarray(ex[2]))
  return c


def argmin(x, axis=None):
  '''
  Compute argmin over ``axis``.

  See `numpy.ndarray.argmin`.

  :param x: `Expr` to compute a minimum over.
  :param axis: Axis (integer or None).
  '''
  compute_min = min(x, axis)
  if axis is not None:
    shape = list(x.shape)
    shape[axis] = 1
    compute_min = compute_min.reshape(tuple(shape))
  argument = map_with_location((x, compute_min), _arg_mapper,
                               fn_kw={'axis': axis})
  return min(argument, axis)


def argmax(x, axis=None):
  '''
  Compute argmax over ``axis``.

  See `numpy.ndarray.argmax`.

  :param x: `Expr` to compute a maximum over.
  :param axis: Axis (integer or None).
  '''
  compute_max = max(x, axis)
  if axis is not None:
    shape = list(x.shape)
    shape[axis] = 1
    compute_max = compute_max.reshape(tuple(shape))
  argument = map_with_location((x, compute_max), _arg_mapper,
                               fn_kw={'axis': axis})
  return min(argument, axis)


def _countnonzero_local(ex, data, axis):
  if axis is None:
    if sp.issparse(data):
      return np.asarray(data.nnz)
    else:
      return np.asarray(np.count_nonzero(data))

  return (data > 0).sum(axis)


def count_nonzero(array, axis=None, tile_hint=None):
  '''
  Return the number of nonzero values in the axis of the ``array``.

  :param array: DistArray or `Expr`.
  :param axis: the axis to count
  :param tile_hint:
  :rtype: np.int64

  '''
  return reduce(array, axis,
                dtype_fn=lambda input: np.int64,
                local_reduce_fn=_countnonzero_local,
                accumulate_fn=np.add,
                tile_hint=tile_hint)


def _countzero_local(ex, data, axis):
  if axis is None:
    return np.asarray(np.prod(ex.shape) - np.count_nonzero(data))

  return (data == 0).sum(axis)


def count_zero(array, axis=None):
  '''
  Return the number of zero values in the axis of the ``array``.

  :param array: DistArray or `Expr`.
  :param axis: the axis to count
  :rtype: np.int64

  '''
  return reduce(array, axis,
                dtype_fn=lambda input: np.int64,
                local_reduce_fn=_countzero_local,
                accumulate_fn=np.add)


def size(x, axis=None):
  '''
  Return the size (product of the size of all axes) of ``x``.

  See `numpy.ndarray.size`.

  :param x: `Expr` to compute the size of.
  '''
  if axis is None:
    return np.prod(x.shape)
  return x.shape[axis]


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


def multiply(a, b):
  assert a.shape == b.shape
  return map((a, b), fn=lambda a, b: a.multiply(b) if sp.issparse(a) else a * b)


def power(a, b):
  return map((a, b), fn=np.power)


def add(a, b):
  return map((a, b), fn=np.add)


def sub(a, b):
  return map((a, b), fn=np.subtract)


def maximum(a, b):
  return map((a, b), np.maximum)


def ln(v):
  return map(v, fn=np.log)


def log(v):
  return map(v, fn=np.log)


def exp(v):
  return map(v, fn=np.exp)


def square(v):
  return map(v, fn=np.square)


def sqrt(v):
  return map(v, fn=np.sqrt)


def abs(v):
  return map(v, fn=np.abs)


def _bincount_mapper(ex, tiles, minlength=None):
  if len(tiles) > 1:
    result = np.bincount(tiles[0], weights=tiles[1], minlength=minlength)
  else:
    result = np.bincount(tiles[0], minlength=minlength)
  result_ex = extent.from_shape(result.shape)
  yield result_ex, result


def bincount(v, weights=None, minlength=None):
  '''
  Count unique values in ``v``.
  See `numpy.bincount` for more information.

  Arguments:
    v (Expr): Array of non-negative integers
  Returns:
    Expr: Integer array of counts.
  '''
  minval = min(v).glom()
  maxval = max(v).glom()
  assert minval > 0
  if minlength is not None:
    minlength = __builtin__.max(maxval + 1, minlength)
  else:
    minlength = maxval + 1

  if weights is not None:
    return map2((v, weights), fn=_bincount_mapper, fn_kw={'minlength': minlength},
                shape=(minlength,), reducer=np.add)
  else:
    return map2(v, fn=_bincount_mapper, fn_kw={'minlength': minlength},
                shape=(minlength,), reducer=np.add)


def _translate_extent(ex, a, roffset=0, coffset=0):
  '''Translate the extent ex into a new extent into a.'''
  offsets = (roffset, coffset)
  ul = [0] * len(ex.ul)
  lr = [0] * len(ex.lr)
  for index in range(len(ul)):
    tmp_ul = ex.ul[index] - offsets[index]
    tmp_lr = ex.lr[index] - offsets[index]
    if tmp_ul >= a.shape[index] or tmp_lr < 0:
      return None
    if tmp_ul < 0:
      tmp_ul = 0
    if tmp_lr > a.shape[index]:
      tmp_lr = a.shape[index]

    ul[index], lr[index] = tmp_ul, tmp_lr

  return extent.create(ul, lr, a.shape)


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


try:
  import scipy.stats

  def norm_cdf(v):
    return map(v, fn=scipy.stats.norm.cdf, numpy_expr='mathlib.norm_cdf')
except:
  print >>sys.stderr, 'Missing scipy.stats (some functions will be unavailable.'
