#!/usr/bin/env python

import collections
from spartan import util
from spartan.util import Assert
import numpy as np
cimport numpy as np

cimport cython
cimport cextent
from libcpp cimport vector

# Can't understand following declaration errors
# Following line makes parakeet with old cython (0.15) report error
#ctypedef unsigned int coordinate_t
# Following line makes parakeet report error
#ctypedef unsigned long long coordinate_t

# Hopfully, 32-dimension is enough.
cdef enum:
  MAX_NDIM=32

cdef class TileExtent(object):
  '''A rectangular tile of a distributed array.
  
  These correspond (roughly) to a `slice` taken from an array
  (without any step component).
  
  Arrays are indexed from the upper-left; for an array of shape
  (sx, sy, sz): (0,0...) is the upper-left corner of an array, 
  and (sx,sy,sz...) the lower-right.
  
  Extents are represented by an upper-left corner (inclusive) and
  a lower right corner (exclusive): [ul, lr).  In addition, they
  carry the shape of the array they are a part of; this is used to
  compute global position information.
  '''
  cdef cextent.CExtent *_cextent

  def __dealloc__(self):
    del self._cextent
    
  def get_ul(self):
    return tuple([self._cextent.ul[i] for i in range(self._cextent.ndim)])

  def set_ul(self, tuple ul):
    for i in range(self._cextent.ndim):
      self._cextent.ul[i] = ul[i]
    self._cextent.init_info()

  def get_lr(self):
    return tuple([self._cextent.lr[i] for i in range(self._cextent.ndim)])

  def set_lr(self, tuple lr):
    for i in range(self._cextent.ndim):
      self._cextent.lr[i] = lr[i]
    self._cextent.init_info()

  def get_array_shape(self):
    if not self._cextent.has_array_shape:
      return None
    return tuple([self._cextent.array_shape[i] for i in range(self._cextent.ndim)])

  def set_array_shape(self, tuple array_shape):
    if not self._cextent.has_array_shape:
      raise NotImplementedError
    for i in range(self._cextent.ndim):
      self._cextent.array_shape[i] = array_shape[i]

  ul = property(get_ul, set_ul)
  lr = property(get_lr, set_lr)
  array_shape = property(get_array_shape, set_array_shape)

  @property
  def size(self):
    return self._cextent.size
  
  @property
  def shape(self):
    return tuple([self._cextent.shape[i] for i in range(self._cextent.ndim)])
  
  @property
  def ndim(self):
    return self._cextent.ndim

  def __reduce__(self):
    return create, (self.ul, self.lr, self.array_shape)
  
  def to_slice(self):
    result = []
    for i in range(self.ndim):
      result.append(slice(self.ul[i], self.lr[i], None))
    return tuple(result)
  
  def __repr__(self):
    return 'extent(' + ','.join('%s:%s' % (a, b) for a, b in zip(self.ul, self.lr)) + ')'
  
  def __getitem__(self, idx):
    return create((self.ul[idx],),
                  (self.lr[idx],),
                  (self.array_shape[idx],))

  def __hash__(self):
    return hash(self.ul)
    
  def __richcmp__(self, other, operation):
    if operation == 0 or operation == 4: # smaller or bigger
      smaller = True
      for i in range(len(self.ul)):
        if self.ul[i] < other.ul[i]:
           smaller = True
           break
        elif self.ul[i] > other.ul[i]:
           smaller = False
           break
      return smaller if operation == 0 else (not smaller)
    elif operation == 2: # eq
      return isinstance(other, TileExtent) and \
             self.ul == other.ul and  \
             self.lr == other.lr
    elif operation == 3: # not eq
      return not isinstance(other, TileExtent) or \
             self.ul != other.ul or \
             self.lr != other.lr
    else:
      assert False, 'Unsupported comparison operation %d' % operation

  def ravelled_pos(self):
    return ravelled_pos(self.ul, self.array_shape)
  
  def to_global(self, idx, axis):
    '''Convert ``idx`` from a local offset in this tile to a global offset.'''
    if axis is not None:
      return idx + self.ul[axis]

    local_idx = unravelled_pos(idx, self.shape)
    return ravelled_pos(np.asarray(self.ul) + local_idx, self.array_shape)

  def add_dim(self):
    #util.log_info('ul:%s lr:%s array_shape:%s', self.ul + (0,), self.lr + (1,), self.array_shape + (1,))
    return create(self.ul + (0,),
                  self.lr + (1,),
                  self.array_shape + (1,))

  def clone(self):
    return c_create(self._cextent.ul, self._cextent.lr, self._cextent.array_shape, self.ndim)
 
#import traceback
counts = collections.defaultdict(int)

from time import time
cdef c_create(unsigned long long *ul, 
              unsigned long long *lr, 
              unsigned long long *array_shape, 
              unsigned int ndim):
  cdef TileExtent ex = TileExtent()
  
  ex._cextent = cextent.extent_create(ul, lr, array_shape, ndim)
  
  if ex._cextent is NULL:
    return None
  return ex

cpdef create(ul, lr, array_shape):
  '''
  Create a new extent with the given coordinates and array shape.
  
  :param ul: `tuple`: 
  :param lr:
  :param array_shape:
  '''
  cdef ndim = len(ul)
  cdef unsigned long long ul_mem[MAX_NDIM]
  cdef unsigned long long lr_mem[MAX_NDIM]
  cdef unsigned long long array_shape_mem[MAX_NDIM]

  for i in xrange(ndim):
    ul_mem[i] = ul[i]
    lr_mem[i] = lr[i]

  if array_shape is not None:
    for i in xrange(ndim):
      array_shape_mem[i] = array_shape[i]
  
  if array_shape is None:
    return c_create(ul_mem, lr_mem, NULL, ndim)
  else:
    return c_create(ul_mem, lr_mem, array_shape_mem, ndim)

def from_shape(shape):
  cdef TileExtent ex = TileExtent()
  cdef ndim = len(shape)
  cdef unsigned long long shape_mem[MAX_NDIM]

  for i in range(ndim):
    shape_mem[i] = shape[i]
  ex._cextent = cextent.extent_from_shape(shape_mem, ndim)
  return ex

@cython.cdivision(True)
cpdef unravelled_pos(idx, array_shape): 
  '''
  Unravel ``idx`` into an index into an array of shape ``array_shape``.
  :param idx: `int`
  :param array_shape: `tuple`
  :rtype: `tuple` indexing into ``array_shape``
  '''
  
  unravelled = []
  for dim in reversed(array_shape):
    unravelled.append(idx % dim)
    idx /= dim
  
  return tuple(reversed(unravelled))
    
cpdef ravelled_pos(idx, array_shape):
  rpos = 0
  mul = 1
  
  for i in range(len(array_shape) - 1, -1, -1):
    rpos += mul * idx[i]
    mul *= array_shape[i]
  
  return rpos

@cython.boundscheck(False)
def all_nonzero_shape(shape):
  '''
  Check if the shape is valid (all elements are biger than zero). This is equal to
  np.all(shape) but is faster because this API doesn't create a numpy array.
  '''
  cdef unsigned int i
  for i in shape:
    if i == 0:
      return False
  return True

@cython.cdivision(True)
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
      
def compute_slice(TileExtent base, idx):
  '''Return a new ``TileExtent`` representing ``base[idx]``
  
  :param base: `TileExtent`
  :param idx: int, slice, or tuple(slice,...)
  '''
  cdef long long slices[MAX_NDIM * 2]

  if not isinstance(idx, tuple):
    idx = (idx,)

  if len(idx) < base.ndim:
    idx = tuple(list(idx) + [slice(None, None, None) 
                             for _ in range(base.ndim - len(idx))])
 
  for i in range(base.ndim):
    slc = idx[i]
    if np.isscalar(slc):
      slices[i * 2] = slc
      slices[i * 2 + 1] = slc + 1
    else:
      slices[i * 2] = int(slc.start) if slc.start is not None else 0
      slices[i * 2 + 1] = int(slc.stop) if slc.stop is not None else int(base.shape[i])

  cdef TileExtent ex = TileExtent()
  
  ex._cextent = cextent.compute_slice_cy(base._cextent, slices, base.ndim)

  return ex

def offset_from(TileExtent base, TileExtent other):
  '''
  :param base: `TileExtent` to use as basis
  :param other: `TileExtent` into the same array.
  :rtype: A new extent using this extent as a basis, instead of (0,0,0...) 
  '''
  Assert.eq(base.array_shape, other.array_shape, 'Tiles must have compatible shapes!')
  cdef TileExtent ex = TileExtent()
  ex._cextent = cextent.offset_from(base._cextent, other._cextent)
  return ex

cpdef offset_slice(TileExtent base, TileExtent other):
  '''
  :param base: `TileExtent` to use as basis
  :param other: `TileExtent` into the same array.
  :rtype: A slice representing the local offsets of ``other`` into this tile.
  '''
  return tuple([slice(other.ul[i] - base.ul[i],
                       other.lr[i] - base.ul[i],
                       None) for i in range(base.ndim)])

def from_slice(idx, shape):
  '''
  Construct a `TileExtent` from a slice or tuple of slices.
  
  :param idx: int, slice, or tuple(slice...)
  :param shape: shape of the input array
  :rtype: `TileExtent` corresponding to ``idx``.
  '''
  if not isinstance(idx, tuple):
    idx = (idx,)
  
  if len(idx) < len(shape):
    idx = tuple(list(idx) + [slice(None, None, None) 
                             for _ in range(len(shape) - len(idx))])
    
  cdef unsigned long long shape_c[MAX_NDIM] 
  cdef long long slices[MAX_NDIM * 2]
  cdef unsigned int ndim, i
 
  ndim = len(shape)
  for i in range(ndim):
    shape_c[i] = shape[i]
    slc = idx[i]

    if np.isscalar(slc):
      slc = int(slc)
      slices[i * 2] = slc
      slices[i * 2 + 1] = slc + 1
    else:
      slices[i * 2] = int(slc.start) if slc.start is not None else 0
      slices[i * 2 + 1] = int(slc.stop) if slc.stop is not None else shape_c[i]

  cdef TileExtent ex = TileExtent()
  ex._cextent = cextent.from_slice_cy(slices, shape_c, ndim)

  return ex

cpdef intersection(TileExtent a, TileExtent b):
  '''
  :rtype: The intersection of the 2 extents as a `TileExtent`, 
          or None if the intersection is empty.  
  '''
  if a is None:
    return None

  Assert.eq(a.array_shape, b.array_shape, 'Tiles must have compatible shapes!')

  cdef TileExtent ex = TileExtent()
  ex._cextent = cextent.intersection(a._cextent, b._cextent)
  if ex._cextent is NULL:
    return None
  return ex


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


def shapes_match(offset, data):
  '''
  Return true if the shape of ``data`` matches the extent ``offset``. 
  :param offset:
  :param data:
  '''
  return np.all(offset.shape == data.shape)

def drop_axis(TileExtent ex, axis):
  if axis is None: return create((), (), ())
  cdef TileExtent ret_ex = TileExtent()

  ret_ex._cextent = cextent.drop_axis(ex._cextent, axis)
  return ret_ex
 
def index_for_reduction(index, axis):
  return drop_axis(index, axis)
        
def find_shape(extents):
  '''
  Given a list of extents, return the shape of the array
  necessary to fit all of them.
  :param extents:
  '''
  #util.log_info('Finding shape... %s', extents)
  shape = np.max([ex.lr for ex in extents], axis=0)
  shape[shape == 0] = 1
  return tuple(shape)

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
