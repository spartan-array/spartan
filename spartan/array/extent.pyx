#!/usr/bin/env python

import collections
from spartan import util
from spartan.util import Assert
import numpy as np

cimport cython
# Hopfully, 32-dimension is enough.
# Parakeet can't convert numpy.int64
#ctypedef unsigned long long[32] coordinate_t
ctypedef unsigned[32] coordinate_t

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
  cdef public tuple array_shape
  cdef coordinate_t c_ul
  cdef coordinate_t c_lr
  cdef unsigned c_ul_len, c_lr_len

  def get_ul(self):
    return tuple([self.c_ul[i] for i in range(self.c_ul_len)])

  def set_ul(self, tuple ul):
    self.c_ul_len = len(ul)
    for i in range(self.c_ul_len):
      self.c_ul[i] = ul[i]

  def get_lr(self):
    return tuple([self.c_lr[i] for i in range(self.c_lr_len)])

  def set_lr(self, tuple _lr):
    self.c_lr_len = len(_lr)
    for i in range(self.c_lr_len):
      self.c_lr[i] = _lr[i]

  ul = property(get_ul, set_ul)
  lr = property(get_lr, set_lr)

  @property
  def size(self):
    return np.prod(self.shape)
  
  #@property
  #def shape(self):
    #result = np.asarray(self.lr) - np.asarray(self.ul)
    #result[result == 0] = 1
    ##util.log_info('Shape: %s', result)
    #return tuple(result)

  @property
  def shape(self):
    result = []
    for i in range(self.c_ul_len):
      result.append(self.c_lr[i] - self.c_ul[i])
      result[i] = 1 if result[i] == 0 else result[i]
    return tuple(result)
  
  @property
  def ndim(self):
    return self.c_ul_len

  def __reduce__(self):
    return create, (self.ul, self.lr, self.array_shape)
  
  def to_slice(self):
    result = []
    for i in range(self.c_ul_len):
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
    if operation == 2: # eq
      return isinstance(other, TileExtent) and \
             self.ul == other.ul and  \
             self.lr == other.lr
    elif operation == 3: # not eq
      return not isinstance(other, TileExtent) or \
             self.ul != other.ul or \
             self.lr != other.lr

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
    self.c_ul[self.c_ul_len] = 0
    self.c_lr[self.c_lr_len] = 1
    return c_create(self.c_ul, 
                    self.c_lr, 
                    self.array_shape + (1,), 
                    self.c_ul_len + 1)

  def clone(self):
    return c_create(self.c_ul, self.c_lr, self.array_shape, self.c_ul_len)
 
#import traceback
counts = collections.defaultdict(int)

cdef c_create(coordinate_t ul, coordinate_t lr, array_shape, unsigned ul_len):
  cdef TileExtent ex = TileExtent()

  # If we got an unrealistic (ul, lr), return None.
  cdef unsigned none = 0 
  with nogil:
    ex.c_ul_len = ex.c_lr_len = ul_len
    for idx in range(ex.c_ul_len):
      if ul[idx] >= lr[idx]:
        none = 1 
      ex.c_ul[idx] = ul[idx]
      ex.c_lr[idx] = lr[idx]
  if none == 1:
    return  None

  if array_shape is not None:
    ex.array_shape = tuple(array_shape)
  else:
    ex.array_shape = None
  
  return ex
  
cpdef create(ul, lr, array_shape):
  '''
  Create a new extent with the given coordinates and array shape.
  
  :param ul: `tuple`: 
  :param lr:
  :param array_shape:
  '''
  cdef TileExtent ex = TileExtent()

  # If we got an unrealistic (ul, lr), return None.
  ex.c_ul_len = ex.c_lr_len = len(ul)
  for idx in range(ex.c_ul_len):
    if ul[idx] >= lr[idx]:
      return None
    ex.c_ul[idx] = ul[idx]
    ex.c_lr[idx] = lr[idx]

  if array_shape is not None:
    ex.array_shape = tuple(array_shape)
  else:
    ex.array_shape = None
  
  return ex

#def from_shape(shp):
  #return create(tuple([0] * len(shp)), tuple(v for v in shp), shp)
def from_shape(shp):
  cdef coordinate_t ul
  cdef coordinate_t lr
  cdef unsigned ul_len, i

  ul_len = len(shp)
  for i in range(ul_len):
    ul[i] = 0
    lr[i] = shp[i]
  return c_create(ul, lr, shp, ul_len)

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
      
#def compute_slice(base, idx):
  #'''Return a new ``TileExtent`` representing ``base[idx]``
  
  #:param base: `TileExtent`
  #:param idx: int, slice, or tuple(slice,...)
  #'''
  #assert not np.isscalar(idx), idx
  #if not isinstance(idx, tuple):
    #idx = (idx,)
    
  #ul = []
  #lr = []
  #array_shape = base.array_shape
  
  #for i in range(len(base.ul)):
    #if i >= len(idx):
      #ul.append(base.ul[i])
      #lr.append(base.lr[i])
    #else:
      #start, stop, step = idx[i].indices(base.shape[i])
      #ul.append(base.ul[i] + start)
      #lr.append(base.ul[i] + stop)
  
  #return create(ul, lr, array_shape)
def compute_slice(TileExtent base, idx):
  '''Return a new ``TileExtent`` representing ``base[idx]``
  
  :param base: `TileExtent`
  :param idx: int, slice, or tuple(slice,...)
  '''
  assert not np.isscalar(idx), idx
  if not isinstance(idx, tuple):
    idx = (idx,)
    
  cdef coordinate_t ul
  cdef coordinate_t lr
  cdef unsigned i

  array_shape = base.array_shape
  for i in range(base.c_ul_len):
    if i >= len(idx):
      ul[i] = base.ul[i]
      lr[i] = base.lr[i]
    else:
      start, stop, step = idx[i].indices(base.shape[i])
      ul[i] = base.ul[i] + start
      lr[i] = base.ul[i] + stop
  
  return c_create(ul, lr, array_shape, base.c_ul_len)

#def offset_from(base, other):
  #'''
  #:param base: `TileExtent` to use as basis
  #:param other: `TileExtent` into the same array.
  #:rtype: A new extent using this extent as a basis, instead of (0,0,0...) 
  #'''
  #assert np.all(other.ul >= base.ul), (other, base)
  #assert np.all(other.lr <= base.lr), (other, base)
  #return create(tuple(np.array(other.ul) - np.array(base.ul)),
                #tuple(np.array(other.lr) - np.array(base.ul)),
                #other.array_shape)
def offset_from(TileExtent base, TileExtent other):
  '''
  :param base: `TileExtent` to use as basis
  :param other: `TileExtent` into the same array.
  :rtype: A new extent using this extent as a basis, instead of (0,0,0...) 
  '''
  cdef coordinate_t ul
  cdef coordinate_t lr
  cdef unsigned i

  for i in range(base.c_ul_len):
    if (other.c_ul[i] < base.ul[i]) or (other.c_lr[i] > base.lr[i]):
      assert False
    ul[i] = other.c_ul[i] - base.c_ul[i]
    lr[i] = other.c_lr[i] - base.c_ul[i]

  return c_create(ul, lr, other.array_shape, base.c_ul_len)

#@cython.boundscheck(False) # turn of bounds-checking for entire function
#cpdef _offset_slice(tuple base_ul,
                 #tuple base_lr,
                 #tuple other_ul,
                 #tuple other_lr):
  #return tuple([slice(ul-base, lr-base, None) for ul, lr, base in zip(other_ul, other_lr, base_ul)])
cpdef offset_slice(TileExtent base, TileExtent other):
  '''
  :param base: `TileExtent` to use as basis
  :param other: `TileExtent` into the same array.
  :rtype: A slice representing the local offsets of ``other`` into this tile.
  '''
  return tuple([slice(other.c_ul[i] - base.c_ul[i], 
                       other.c_lr[i] - base.c_ul[i], 
                       None) for i in range(base.c_ul_len)])
  #return _offset_slice(base.ul, base.lr, other.ul, other.lr)

#def from_slice(idx, shape):
  #'''
  #Construct a `TileExtent` from a slice or tuple of slices.
  
  #:param idx: int, slice, or tuple(slice...)
  #:param shape: shape of the input array
  #:rtype: `TileExtent` corresponding to ``idx``.
  #'''
  #if not isinstance(idx, tuple):
    #idx = (idx,)
  
  #if len(idx) < len(shape):
    #idx = tuple(list(idx) + [slice(None, None, None) 
                             #for _ in range(len(shape) - len(idx))])
    
  #ul = []
  #lr = []
 
  #for i in range(len(shape)):
    #dim = shape[i]
    #slc = idx[i]
    
    #if np.isscalar(slc):
      #slc = int(slc)
      #slc = slice(slc, slc + 1, None)
    
    #if slc.start > 0: assert slc.start <= dim
    #if slc.stop > 0: assert slc.stop <= dim
    
    #indices = slc.indices(dim)
    #ul.append(indices[0])
    #lr.append(indices[1])
    
  #return create(tuple(ul), tuple(lr), shape)
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
    
  cdef coordinate_t ul
  cdef coordinate_t lr 
  cdef unsigned ul_len, i
 
  ul_len = len(shape)
  for i in range(ul_len):
    dim = shape[i]
    slc = idx[i]
    
    if np.isscalar(slc):
      slc = int(slc)
      slc = slice(slc, slc + 1, None)
    
    if slc.start > 0: assert slc.start <= dim
    if slc.stop > 0: assert slc.stop <= dim
    
    indices = slc.indices(dim)
    ul[i] = indices[0]
    lr[i] = indices[1]
    
  return c_create(ul, lr, shape, ul_len)

#def intersection(a, b):
  #'''
  #:rtype: The intersection of the 2 extents as a `TileExtent`, 
          #or None if the intersection is empty.  
  #'''
  #if a is None:
    #return None
  
  #for i in range(len(a.lr)):
    #if b.lr[i] < a.ul[i]: return None
    #if a.lr[i] < b.ul[i]: return None
    
  #Assert.eq(a.array_shape, b.array_shape)
  
  #return create(np.maximum(b.ul, a.ul),
                #np.minimum(b.lr, a.lr),
                #a.array_shape)
cpdef intersection(TileExtent a, TileExtent b):
  '''
  :rtype: The intersection of the 2 extents as a `TileExtent`, 
          or None if the intersection is empty.  
  '''
  if a is None:
    return None
  
  cdef coordinate_t ul
  cdef coordinate_t lr
  cdef unsigned i

  for i in range(a.c_ul_len):
    if b.c_lr[i] < a.c_ul[i]: return None
    if a.c_lr[i] < b.c_ul[i]: return None
    ul[i] = a.c_ul[i] if a.c_ul[i] >= b.c_ul[i] else b.c_ul[i]
    lr[i] = a.c_lr[i] if a.c_lr[i] <  b.c_lr[i] else b.c_lr[i]
    
  Assert.eq(a.array_shape, b.array_shape)
  
  return c_create(ul, lr, a.array_shape, a.c_ul_len)


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

#def drop_axis(ex, axis):
  #if axis is None: return create((), (), ())
  #if axis < 0: axis = len(ex.ul) + axis
  
  #ul = list(ex.ul)
  #lr = list(ex.lr)
  #shape = list(ex.array_shape)
  #del ul[axis]
  #del lr[axis]
  #del shape[axis]
  #return create(ul, lr, shape)
def drop_axis(TileExtent ex, axis):
  if axis is None: return create((), (), ())
  if axis < 0: axis = ex.c_ul_len + axis
  
  cdef coordinate_t ul
  cdef coordinate_t lr
  cdef unsigned i

  shape = list(ex.array_shape)
  del shape[axis]
  for i in range(axis):
    ul[i] = ex.c_ul[i]
    lr[i] = ex.c_lr[i]

  for i in range(axis + 1, ex.c_ul_len):
    ul[i - 1] = ex.c_ul[i]
    lr[i - 1] = ex.c_lr[i]

  return c_create(ul, lr, shape, ex.c_ul_len - 1)
 
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
