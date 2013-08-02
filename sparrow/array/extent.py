import numpy as N


class TileExtent(object):
  '''A rectangular tile of a distributed array.'''
  def __init__(self, ul, sz, array_shape):
    self.ul = N.array(ul)
    self.sz = N.array(sz)
    self.array_shape = N.array(array_shape)
  
  @property
  def lr(self):
    return self.ul + self.sz
  
  @property
  def shape(self):
    return self.sz
  
  def to_slice(self):
    return tuple([slice(ul, ul + sz, None) for ul, sz in zip(self.ul, self.sz)])

  def __repr__(self):
    return ','.join('%s:%s' % (a, b) for a, b in zip(self.ul, self.lr))

  def drop_axis(self, axis):
    if axis is None: return TileExtent((0,), (1,), (1,))
    ul = list(self.ul)
    sz = list(self.sz)
    shape = list(self.array_shape)
    del ul[axis]
    del sz[axis]
    del shape[axis]

#    util.log('%s -> %s, %s -> %s', self.ul, ul, self.sz, sz)
    return TileExtent(ul, sz, shape)

  def __hash__(self):
    return hash((tuple(self.ul), tuple(self.sz)))

  def __eq__(self, other):
    return N.all(self.ul == other.ul) and N.all(self.sz == other.sz)

  def ravelled_pos(self, nd_pos):
    pos = 0
    for i in range(len(self.array_shape) - 1):
      pos += self.array_shape[i] * nd_pos[i]
    return pos + nd_pos[-1]

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
    
    unravelled = N.array(list(reversed(unravelled)))
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
      return self.ravelled_pos(self.ul + self.sz)
    return self.ul[axis] + self.sz[axis]

  def size(self, axis):
    if axis is None:
      return N.prod(self.sz)
    return self.sz[axis]
  
  def local_offset(self, other):
    '''
    :param other: `TileExtent` into the same array.
    :rtype: A tuple of local slices for this tile.
    '''
    assert N.all(other.ul >= self.ul)
    assert N.all(other.sz + other.ul <= self.ul + self.sz)
    return tuple([slice(p, p + s, None) for (p, s) in zip(other.ul - self.ul, other.sz)])
  
  def create_array(self):
    return N.ndarray(self.shape)


def intersection(a, b):
  '''
  :rtype: The intersection of the 2 extents as a `TileExtent`, or None if the intersection is empty.  
  '''
  if N.any(b.lr <= a.ul): return None
  if N.any(a.lr <= b.ul): return None
  return TileExtent(N.maximum(b.ul, a.ul),
                    N.minimum(b.lr, a.lr) - N.maximum(b.ul, a.ul),
                    a.array_shape)

TileExtent.intersection = intersection