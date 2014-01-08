import traceback
import numpy as np
import scipy
from scipy.sparse import dok_matrix
from . import extent
from spartan import util
from spartan.util import Assert

TYPE_EMPTY = 0
TYPE_DENSE = 1
TYPE_MASKED = 2
TYPE_SPARSE = 3

MASK_ALL_CLEAR = 0
MASK_ALL_SET = 1

ID = iter(xrange(100000000))

class Tile(object):
  '''
  Tiles have 4 modes:

  Empty   -- no data or mask
  Masked  -- data + mask
  Sparse  -- hashmap of positions (implicit mask)
  Dense   -- all data values have been set, mask is cleared.
  '''

  def __init__(self, shape, dtype, data, mask, tile_type):
    Assert.ne(dtype, object)
    self.id = ID.next()
    self.shape = shape
    self.dtype = dtype
    self.type = tile_type
    self.mask = mask
    self.data = data

    if data is not None:
      Assert.eq(data.shape, shape)
      Assert.eq(data.dtype, dtype)

    if isinstance(mask, np.ndarray):
      Assert.eq(mask.shape, shape)

  @property
  def data(self):
    #util.log_info('DATA %s %s', self.id, self._data)
    return self._data

  @data.setter
  def data(self, val):
    #util.log_info('SET DATA %s %s', self.id, val)
    if val is not None:
      Assert.eq(val.dtype, self.dtype)

    self._data = val

  def update(self, update, reducer):
    #util.log_info('Update: %s %s', update.data, reducer)
    return merge(self, update, reducer)

  def get(self, selector=None):
    # no data, return an empty array
    if self.data is None:
      #util.log_info('EMPTY %s %s', self.id, self.shape)
      return np.ndarray(self.shape, self.dtype)[selector]

    # scalars are just returned directly.
    if len(self.data.shape) == 0:
      return self.data

    Assert.le(len(selector), len(self.data.shape),
              'Selector has more dimensions than data! %s %s' % (selector, self.data.shape))

    # otherwise if sparse, return a sparse subset
    if self.type == TYPE_SPARSE:
      if selector is None or extent.is_complete(self.data.shape, selector):
        result = self.data
      else:
        result = self.data[selector]
      return result
    
    # dense, check our mask and return a masked segment or unmasked if
    # the mask is all filled for the selected region.
    self._initialize_mask()
    if self.mask is None or np.all(self.mask[selector]):
      #util.log_info('%s %s %s', self.data, self.mask, selector)
      return self.data[selector]
      
    data = self.data[selector]
    mask = self.mask[selector]
    result = np.ma.masked_all(data.shape, dtype=data.dtype)
    result[mask] = data[mask]

    #util.log_info('%s %s', self.id, result)
    return result

  def _initialize(self):
    # if this is a scalar, then data is just the scalar value itself.
    if len(self.shape) == 0:
      return

    if self.type == TYPE_SPARSE:
      if self.data is None:
        self.data = dok_matrix(self.shape, self.dtype)
    else:
      self._initialize_mask()

  def _initialize_mask(self):
    if not isinstance(self.mask, np.ndarray):
      if self.mask == MASK_ALL_SET:
        self.mask = np.ones(self.shape, dtype=np.bool)
      else:
        self.mask = np.zeros(self.shape, dtype=np.bool)


  def __repr__(self):
    return 'tile(%s, %s)' % (self.shape, self.dtype)


def from_data(data):
  if isinstance(data, scipy.sparse.spmatrix):
    return Tile(
      shape=data.shape,
      data=data,
      dtype=data.dtype,
      mask=MASK_ALL_SET,
      tile_type=TYPE_SPARSE)
  else:
    return Tile(
      shape=data.shape,
      data=data,
      dtype=data.dtype,
      mask=MASK_ALL_SET,
      tile_type=TYPE_DENSE)


def from_shape(shape, dtype, tile_type=TYPE_DENSE):
  assert tile_type == TYPE_DENSE
  return Tile(shape=shape,
              data=None,
              dtype=dtype,
              tile_type=tile_type,
              mask=MASK_ALL_CLEAR)


def from_intersection(src, overlap, data):
  '''
  Return a tile for ``src``, masked to update the area specifed by ``overlap``.
  
  :param src: `TileExtent`
  :param overlap: `TileExtent`
  :param data:
  '''
  util.log_debug('%s %s %s', src, overlap, data.dtype)
  slc = extent.offset_slice(src, overlap)
  tdata = np.ndarray(src.shape, data.dtype)
  tmask = np.zeros(src.shape, np.bool)
  tmask[slc] = 1
  tdata[slc] = data
  return Tile(dtype=data.dtype,
              data=tdata,
              shape=src.shape,
              mask=tmask,
              tile_type=TYPE_DENSE)


def merge(old_tile, new_tile, reducer):
  Assert.isinstance(old_tile, Tile)
  Assert.isinstance(new_tile, Tile)

  Assert.eq(old_tile.dtype, new_tile.dtype)
  Assert.eq(old_tile.shape, new_tile.shape)

  # fast path -- replace an empty tile with the new data
  if old_tile.data is None:
    #util.log_info('REPL %s %s', new_tile.id, new_tile.data)
    return new_tile

  #util.log_info('OLD: %s %s', old_tile.id, old_tile.data)
  #util.log_info('NEW: %s %s', new_tile.id, new_tile.data)

  # zero-dimensional arrays; just use data == None as a mask.
  if len(old_tile.shape) == 0:
    if old_tile.data is None or reducer is None:
      old_tile.data = new_tile.data
    else:
      old_tile.data = reducer(old_tile.data, new_tile.data)
    return old_tile

  old_tile._initialize()
  new_tile._initialize()

  Assert.eq(old_tile.shape, new_tile.shape)
  if old_tile.type == TYPE_DENSE:
    replaced = ~old_tile.mask & new_tile.mask
    updated = old_tile.mask & new_tile.mask

    #util.log_info('%s %s', old_tile.mask, new_tile.mask)
    #util.log_info('REPLACE: %s', replaced)
    #util.log_info('UPDATE: %s', updated)

    old_tile.data[replaced] = new_tile.data[replaced]
    old_tile.mask[replaced] = 1

    if np.any(updated):
      if reducer is None:
        old_tile.data[updated] = new_tile.data[updated]
      else:
        old_tile.data[updated] = reducer(old_tile.data[updated], new_tile.data[updated])
  else:
    # sparse update, no mask, just iterate over data items
    # TODO(POWER) -- this is SLOW!
    for k, v in new_tile.iteritems():
      if old_tile.has_key(k):
        old_tile[k] = reducer(old_tile[k], v)
      else:
        old_tile[k] = v

  return old_tile
