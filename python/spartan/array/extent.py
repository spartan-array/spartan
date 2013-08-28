from spartan import util
import numpy as np

class TileExtent(object):
  '''A rectangular tile of a distributed array.'''
  def __init__(self, ul, sz, array_shape):
    self.ul = tuple(ul)
    self.sz = tuple(sz)
    self.array_shape = tuple(array_shape)
    self.shape = self.sz
    
    # cache some values as numpy arrays for faster access
    self.ul_array = np.asarray(self.ul, dtype=np.int)
    self.sz_array = np.asarray(self.sz, dtype=np.int)
    self.lr_array = self.ul_array + self.sz_array
    
    self.lr = tuple(self.lr_array)
    
  def __reduce__(self):
    return (TileExtent, (self.ul, self.sz, self.array_shape))
  
  @property
  def ndim(self):
    return len(self.sz)
  
  def to_slice(self):
    return tuple([slice(ul, lr, None) for ul, lr in zip(self.ul, self.lr)])

  def __repr__(self):
    return 'extent(' + ','.join('%s:%s' % (a, b) for a, b in zip(self.ul, self.lr)) + ')'

  def drop_axis(self, axis):
    if axis is None: return TileExtent((), (), ())
    ul = list(self.ul)
    sz = list(self.sz)
    shape = list(self.array_shape)
    del ul[axis]
    del sz[axis]
    del shape[axis]

#    util.log('%s -> %s, %s -> %s', self.ul, ul, self.sz, sz)
    return TileExtent(ul, sz, shape)

  def __hash__(self):
    return hash(tuple(self.ul)) ^ hash(tuple(self.sz))

  def __eq__(self, other):
    return np.all(self.ul == other.ul) and np.all(self.sz == other.sz)

  def ravelled_pos(self, global_pos=None):
    if global_pos is None:
      global_pos = self.ul
    
    pos = 0
    for i in range(len(self.array_shape) - 1):
      pos += self.array_shape[i] * global_pos[i]
    return pos + global_pos[-1]
  
  def to_global(self, idx, axis):
    '''Convert ``idx`` from a local offset in this tile to a global offset.'''
    if axis is not None:
      return idx + self.ul[axis]

    # first unravel idx to a local position
    local_idx = idx
    unravelled = []
    for i in range(len(self.sz)):
      unravelled.append(local_idx % self.sz[i])
      local_idx /= self.sz[i]
    
    unravelled = np.array(list(reversed(unravelled)))
    unravelled += self.ul
#    util.log('%s, %s, %s, %s %s',
#             self.ul, idx, unravelled, self.ravelled_pos(unravelled), self.array_shape)
    return self.ravelled_pos(unravelled)

  def start(self, axis):
    if axis is None:
      return self.ravelled_pos(self.ul)
    return self.ul[axis]

  def stop(self, axis):
    if axis is None:
      return self.ravelled_pos(self.lr)
    return self.ul[axis] + self.sz[axis]

  def size(self, axis):
    if axis is None:
      return np.prod(self.sz)
    return self.sz[axis]
  
  def create_array(self):
    return np.ndarray(self.shape)
  

def offset_from(base, other):
  '''
  :param base: `TileExtent` to use as basis
  :param other: `TileExtent` into the same array.
  :rtype: A new extent using this extent as a basis, instead of (0,0,0...) 
  '''
  assert np.all(other.ul >= base.ul)
  assert np.all(other.lr <= base.lr)
  return TileExtent(np.array(other.ul) - np.array(base.ul), other.sz, other.shape)

def offset_slice(base, other):
  '''
  :param base: `TileExtent` to use as basis
  :param other: `TileExtent` into the same array.
  :rtype: A slice representing the local offsets of ``other`` into this tile.
  '''
  return offset_from(base, other).to_slice()
  #return tuple([slice(p, p + s, None) for (p, s) in zip(other.ul - self.ul, other.sz)])
  

def from_slice(idx, shape):
  if np.isscalar(idx):
    idx = int(idx)
    idx = slice(idx, idx + 1, None)
  if not isinstance(idx, tuple):
    idx = (idx,)
  if len(idx) < len(shape):
    idx = tuple(list(idx) + [slice(None, None, None) 
                             for _ in range(len(shape) - len(idx))])
    
  ul = []
  sz = []
  
  for i in range(len(shape)):
    dim = shape[i]
    slc = idx[i]
    indices = slc.indices(dim)
    ul.append(indices[0])
    sz.append(indices[1] - indices[0])
    
  return TileExtent(ul, sz, shape)


def intersection(a, b):
  '''
  :rtype: The intersection of the 2 extents as a `TileExtent`, 
          or None if the intersection is empty.  
  '''
  for i in range(len(a.lr)):
    if b.lr[i] < a.ul[i]: return None
    if a.lr[i] < b.ul[i]: return None
  #if np.any(b.lr_array <= a.ul_array): return None
  #if np.any(a.lr_array <= b.ul_array): return None
  return TileExtent(np.maximum(b.ul_array, a.ul_array),
                    np.minimum(b.lr_array, a.lr_array) - 
                    np.maximum(b.ul_array, a.ul_array),
                    a.array_shape)

TileExtent.intersection = intersection


def shape_for_reduction(input_shape, axis):
  if axis == None: return ()
  input_shape = list(input_shape)
  del input_shape[axis]
  return input_shape


def shapes_match(offset, data):
  return np.all(offset.sz == data.shape)

def index_for_reduction(index, axis):
  return index.drop_axis(axis)

def shape_for_slice(input_shape, slc):
  raise NotImplementedError