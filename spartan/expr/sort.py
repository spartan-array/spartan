import numpy as np
from .ndarray import ndarray
from ..array import extent
from .tile_operation import tile_operation
from .shuffle import shuffle
from .base import force
from .. import util
from .. import rpc
from .. import blob_ctx


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


def _sort_mapper(array, ex, axis=None):
  axis_ex = extent.change_partition_axis(ex, axis)
  assert axis_ex is not None, "Spartan doesn't support sort for block partition"
  tile = array.fetch(axis_ex)
  yield axis_ex, np.sort(tile, axis=axis)


def sort(array, axis=-1, sample_rate=0.1):
  '''
  sort the array into a flatten format.

  Args:
    array(DistArray or Expr): array to be sorted.
    sample_rate(float): the sample rate.
  '''
  if axis is not None:
    return shuffle(array, _sort_mapper, kw={'axis': axis},
                   shape_hint=array.shape)

  array = force(array)

  # sample the original array
  local_sorted_array = ndarray(array.shape, dtype=array.dtype, tile_hint=array.tile_shape()).force()
  samples = tile_operation(array, fn=_sample_sort_mapper, kw={'sample_rate': sample_rate, 'local_sorted_array': local_sorted_array}).force().values()
  sorted_samples = np.sort(np.concatenate(samples), axis=None)

  # calculate the partition keys, generate the index of each partition key in each tile.
  steps = max(1, sorted_samples.size/len(array.tiles))
  partition_keys = sorted_samples[steps::steps]
  partition_counts = tile_operation(local_sorted_array, fn=_partition_count_mapper, kw={'partition_keys': partition_keys}).force()

  # sort the local_sorted_array into global sorted array
  sorted_array = shuffle(local_sorted_array, fn=_fetch_sort_mapper, kw={'partition_counts': partition_counts}, shape_hint=local_sorted_array.shape)

  return sorted_array


# TODO: Support partition with axis
def partition(array, kth, axis=-1):
  raise NotImplementedError


def _argsort_mapper(array, ex, axis=None):
  axis_ex = extent.change_partition_axis(ex, axis)
  assert axis_ex is not None, "Spartan doesn't support argsort for block partition"
  tile = array.fetch(axis_ex)
  yield axis_ex, np.argsort(tile, axis=axis)


def argsort(array, axis=-1):
  '''
  argsort the array alone axis. If axis is none, the matrix will be flaten.

  Args:
    array(DistArray or Expr): array to be sorted
    axis(int): axis
  '''

  assert axis is not None, "Spartan doesn't support argsort when axis == None now"
  return shuffle(array, _argsort_mapper, kw={'axis': axis},
                 shape_hint=array.shape)


def _argpartition_mapper(array, ex, kth=None, axis=None):
  axis_ex = extent.change_partition_axis(ex, axis)
  assert axis_ex is not None, "Spartan doesn't support argpartition for block partition"
  tile = array.fetch(axis_ex)
  yield axis_ex, np.argpartition(tile, kth, axis=axis)


def argpartition(array, kth, axis=-1):
  '''
  argpartition the array alone axis. If axis is none, the matrix will be flaten.

  Args:
    array(DistArray or Expr): array to be sorted
    kth(int, or ints): Element index to partition by.
    axis(int): axis
  '''
  assert axis is not None, "Spartan doesn't support argpartition when axis == None now"
  return shuffle(array, _argsort_mapper, kw={'axis': axis},
                 shape_hint=array.shape)
