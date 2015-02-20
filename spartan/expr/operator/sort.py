import numpy as np

from .map import map2
from .ndarray import ndarray
from .shuffle import shuffle
from .tile_operation import tile_operation
from ... import util, rpc, blob_ctx
from ...array import extent


def _sample_sort_mapper(array, ex, sample_rate, local_sorted_array):
  '''
  sample each tile of the original array with sample_rate, sort each tile
  and put the local sorted result into local_sorted_array.

  Args:
    array(DistArray): array to be sorted.
    ex(Extent): Region being processed.
    sample_rate(float): the sample rate of each tile.
    local_sorted_array(DistArray): array to store the local sorted result.
  '''
  data = array.fetch(ex)
  samples = np.random.choice(data.flatten(), size=int(data.size * sample_rate), replace=False)
  local_sorted_array.update(ex, np.sort(data, axis=None).reshape(ex.shape))
  yield None, samples


def _partition_count_mapper(array, ex, partition_keys):
  '''
  given the partition keys, calculate the index of each partition key in the local tile.

  Args:
    array(DistArray): local sorted array.
    ex(Extent): Region being processed.
    partition_keys(numpy.array): the partition keys which separate each partitions.
  '''
  data = array.fetch(ex).flatten()
  idx = np.searchsorted(data, partition_keys, side='left')
  idx = np.insert(idx, 0, 0)
  idx = np.append(idx, data.size)
  yield None, idx


def _fetch_sort_mapper(array, ex, partition_counts):
  '''
  given the partition counts, fetch data which belong to this partition from all the tiles and sort them.

  Args:
    array(DistArray): local sorted array.
    ex(Extent): Region being processed.
    partition_counts(dict): the index of each partition key in each tiles. (tile_id -> indices of each partition key)
  '''
  sorted_exts = sorted(array.tiles.keys(), key=lambda x: x.ul)
  id = sorted_exts.index(ex)

  ctx = blob_ctx.get()
  futures = rpc.FutureGroup()
  dst_idx = 0
  for ex in sorted_exts:
    tile_id = array.tiles[ex]
    dst_idx += partition_counts[tile_id][0][id]

    # there are data belong to local partition in the tile ex
    if partition_counts[tile_id][0][id+1] > partition_counts[tile_id][0][id]:
      fetch_slice = tuple([slice(partition_counts[tile_id][0][id], partition_counts[tile_id][0][id+1], None)])
      futures.append(ctx.get_flatten(tile_id, fetch_slice, wait=False))

  result = np.concatenate([resp.data for resp in futures.wait()], axis=None)
  yield extent.create((dst_idx,), (dst_idx+result.size,), (np.prod(array.shape),)), np.sort(result, axis=None)


def _sort_mapper(extents, tiles, axis=None):
  yield extents[0], np.sort(tiles[0], axis=axis)


def sort(array, axis=-1, sample_rate=0.1):
  '''
  sort the array into a flatten format.

  Args:
    array(DistArray or Expr): array to be sorted.
    sample_rate(float): the sample rate.
  '''
  if axis is not None:
    if axis < 0:
      axis = len(array.shape) + axis
    partition_axis = extent.largest_dim_axis(array.shape, exclude_axes=[axis])
    return map2(array, partition_axis, fn=_sort_mapper,
                fn_kw={'axis': axis}, shape=array.shape)

  array = array.evaluate()

  # sample the original array
  local_sorted_array = ndarray(array.shape, dtype=array.dtype, tile_hint=array.tile_shape()).evaluate()
  samples = tile_operation(array, fn=_sample_sort_mapper, kw={'sample_rate': sample_rate, 'local_sorted_array': local_sorted_array}).evaluate().values()
  sorted_samples = np.sort(np.concatenate(samples), axis=None)

  # calculate the partition keys, generate the index of each partition key in each tile.
  steps = max(1, sorted_samples.size/len(array.tiles))
  partition_keys = sorted_samples[steps::steps]
  partition_counts = tile_operation(local_sorted_array, fn=_partition_count_mapper, kw={'partition_keys': partition_keys}).evaluate()

  # sort the local_sorted_array into global sorted array
  sorted_array = shuffle(local_sorted_array, fn=_fetch_sort_mapper, kw={'partition_counts': partition_counts}, shape_hint=local_sorted_array.shape)

  return sorted_array


def _partition_mapper(extents, tiles, axis=None):
  yield extents[0], np.partition(tiles[0], axis=axis)


def partition(array, kth, axis=-1):
  """
  Return a partitioned copy of an array.

  Args: array: DistArray or Expr
    array to be sorted
  kth:  int or list of ints
    Index to partition by
  axis:	int or None, optional
    Axis along which to sort.

  RETURN: ndarray expr
  """
  assert axis is not None, "Spartan doesn't support partition when axis == None now"
  if axis is not None:
    if axis < 0:
      axis = len(array.shape) + axis
    partition_axis = extent.largest_dim_axis(array.shape, exclude_axes=[axis])
    return map2(array, partition_axis, fn=_partition_mapper,
                fn_kw={'axis': axis}, shape=array.shape)


def _argsort_mapper(extents, tiles, axis=None):
  yield extents[0], np.argsort(tiles[0], axis=axis)


def argsort(array, axis=-1):
  '''
  argsort the array alone axis. If axis is none, the matrix will be flaten.

  Args:
    array(DistArray or Expr): array to be sorted
    axis(int): axis
  '''
  assert axis is not None, "Spartan doesn't support argsort when axis == None now"
  if axis is not None:
    if axis < 0:
      axis = len(array.shape) + axis
    partition_axis = extent.largest_dim_axis(array.shape, exclude_axes=[axis])
    return map2(array, partition_axis, fn=_argsort_mapper,
                fn_kw={'axis': axis}, shape=array.shape)


def _argpartition_mapper(extents, tiles, axis=None):
  yield extents[0], np.argpartition(tiles[0], axis=axis)


def argpartition(array, kth, axis=-1):
  '''
  argpartition the array alone axis. If axis is none, the matrix will be flaten.

  Args:
    array(DistArray or Expr): array to be sorted
    kth(int, or ints): Element index to partition by.
    axis(int): axis
  '''
  assert axis is not None, "Spartan doesn't support argpartition when axis == None now"
  if axis is not None:
    if axis < 0:
      axis = len(array.shape) + axis
    partition_axis = extent.largest_dim_axis(array.shape, exclude_axes=[axis])
    return map2(array, partition_axis, fn=_argpartition_mapper,
                fn_kw={'axis': axis}, shape=array.shape)
