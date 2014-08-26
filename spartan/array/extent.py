import numpy as np
from _cextent_py_if import *

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
  if axis == None: return ()
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
    overlap = extent.intersection(ex, region)
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

  for dim,slice in zip(shape, slices):
    if slice.start > 0: return False
    if slice.stop < dim: return False
  return True
