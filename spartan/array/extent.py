import numpy as np
import copy_reg
from _cextent_py_if import *
from spartan import util

#def offset_slice(base, other):
  #'''
  #:param base: `TileExtent` to use as basis
  #:param other: `TileExtent` into the same array.
  #:rtype: A slice representing the local offsets of ``other`` into this tile.
  #'''
  #a = tuple([slice(other.ul[i] - base.ul[i],
                       #other.lr[i] - base.ul[i],
                       #None) for i in range(base.ndim)])
  #print base.ul, other.ul, other.lr
  #print 'offset_slice', a
  #return a;

#def extent_reduce(self):
  #return create, (self.ul, self.lr, self.array_shape)


def extent_reduce(self):
  return create, self.to_tuple()

copy_reg.pickle(TileExtent, extent_reduce)


def to_global(ex, idx, axis):
  '''Convert ``idx`` from a local offset in this tile to a global offset.'''
  if axis is not None:
    return idx + ex.ul[axis]

  rpos = ex.to_global(idx)
  return np.int64(rpos)


def shapes_match(offset, data):
  '''
  Return true if the shape of ``data`` matches the extent ``offset``.
  :param offset:
  :param data:
  '''
  return np.all(offset.shape == data.shape)


def shape_for_reduction(input_shape, axis):
  '''
  Return the shape for the result of applying a reduction along ``axis`` to
  an input of shape ``input_shape``.
  :param input_shape:
  :param axis:
  '''
  if axis is None: return ()
  input_shape = list(input_shape)
  del input_shape[axis]
  return input_shape


def find_overlapping(extents, region):
  '''
  Return the extents that overlap with ``region``.

  :param extents: List of extents to search over.
  :param region: `Extent` to match.
  '''
  for ex in extents:
    overlap = intersection(ex, region)
    if overlap is not None:
      yield (ex, overlap)


def all_nonzero_shape(shape):
  '''
  Check if the shape is valid (all elements are biger than zero). This is equal to
  np.all(shape) but is faster because this API doesn't create a numpy array.
  '''
  for i in shape:
    if i == 0:
      return False
  return True


def find_rect(ravelled_ul, ravelled_lr, shape):
  '''
  Return a new (ravellled_ul, ravelled_lr) to make a rectangle for `shape`.
  If (ravelled_ul, ravelled_lr) already forms a rectangle, just return it.

  :param ravelled_ul:
  :param ravelled_lr:
  '''
  if shape[-1] == 1 or ravelled_ul / shape[-1] == ravelled_lr / shape[-1]:
    rect_ravelled_ul = ravelled_ul
    rect_ravelled_lr = ravelled_lr
  else:
    div = 1
    for i in shape[1:]:
      div = div * i
    rect_ravelled_ul = ravelled_ul - (ravelled_ul % div)
    rect_ravelled_lr = ravelled_lr + (div - ravelled_lr % div) % div - 1
  return (rect_ravelled_ul, rect_ravelled_lr)


def is_complete(shape, slices):
  '''
  Returns true if ``slices`` is a complete covering of shape; that is:

  ::

    array[slices] == array

  :param shape: tuple of int
  :param slices: list/tuple of `slice` objects
  :rtype: boolean
  '''
  if len(shape) != len(slices):
    return False

  for dim, slice in zip(shape, slices):
    if slice.start > 0: return False
    if slice.stop < dim: return False
  return True


def largest_intact_dim_axis(ex, exclude_axes=None):
  '''
  Args:
    shape:
    exclude_axes: tuple or list
  '''
  idx = np.argsort(ex.array_shape)
  for i in xrange(len(idx)-1, -1, -1):
    if ex.shape[idx[i]] == ex.array_shape[idx[i]] and \
        (exclude_axes is None or idx[i] not in exclude_axes):
      return idx[i]
  for i in xrange(len(idx)-1, -1, -1):
    if exclude_axes is None or idx[i] not in exclude_axes:
      return idx[i]


def partition_axes(ex):
  partition_axes = []
  for i in xrange(len(ex.shape)):
    if ex.shape[i] != ex.array_shape[i]:
      partition_axes.append(i)

  return partition_axes


def change_partition_axis(ex, axis):
  if isinstance(axis, (list, tuple)):
    # Changing to grid partition
    old_axes = partition_axes(ex)
    if len(old_axes) > 1:
      # TODO: Changing from a kind of grid partition to another grid partition.
      # May be useful for multi-dimensional matrices.
      return ex
    else:
      # FIXME: We assume that every extent has similar size when calling this
      # API. Otherwise, we don't know how to do repartition.
      # TODO: Our extent may need an extra information which is index.
      old_axis = old_axes[0]
      n_dim = len(axis)
      step = ex.lr[old_axis] - ex.ul[old_axis]
      ntiles = util.divup(ex.array_shape[old_axis], step)
      original_index = int(ex.ul[old_axis] / step)
      n = int(math.pow(ntiles, 1.0 / n_dim))
      grid_index = [0 for i in range(n_dim)]
      for i in reversed(range(n_dim)):
        grid_index[i] = original_index % n
        original_index -= grid_index[i]
        original_index /= n
      steps = [util.divup(ex.array_shape[i], n) for i in range(n_dim)]
      ul = [steps[i] * grid_index[i] for i in range(n_dim)]
      lr = [steps[i] * (grid_index[i] + 1) for i in range(n_dim)]
      for i in range(len(lr)):
        if lr[i] > ex.array_shape[i]:
          return None
      return create(ul, lr, ex.array_shape)
  else:
    # Change to one-dimension partition
    if axis < 0:
      axis += len(ex.array_shape)

    # Vector is a special case
    if len(ex.shape) == 1:
      if axis == 1:
        # We define that if axis is 1, users need the whole vector.
        return create((0, ), ex.array_shape, ex.array_shape)
      else:
        return ex

    old_axes = partition_axes(ex)
    if len(old_axes) > 1:
      # Changing from grid tiling to one-dimensional tiling.
      blk_idx = (ex.ul[0]/ex.shape[0]) * util.divup(ex.array_shape[1], ex.shape[1]) + ex.ul[1]/ex.shape[1]
      ul = [0, 0]
      lr = list(ex.array_shape)
      ul[axis] = blk_idx
      lr[axis] = blk_idx+1
      return create(ul, lr, ex.array_shape)

    if len(old_axes) == 0 or old_axes[0] == axis:
      return ex

    old_axis = old_axes[0]

    new_ul = list(ex.ul[:])
    new_lr = list(ex.lr[:])
    new_ul[axis] = util.divup(new_ul[old_axis] * ex.array_shape[axis],
                              ex.array_shape[old_axis])
    new_ul[old_axis] = 0
    new_lr[axis] = util.divup(new_lr[old_axis] * ex.array_shape[axis],
                              ex.array_shape[old_axis])
    new_lr[old_axis] = ex.array_shape[old_axis]

    target_ex = create(new_ul, new_lr, ex.array_shape)
    #assert target_ex is not None, (new_ul, new_lr, axis, ex.array_shape, ex)
    return target_ex
