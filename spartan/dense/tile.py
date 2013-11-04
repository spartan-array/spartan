import numpy as np

from . import extent
from spartan import util
from spartan.util import Assert


NONE_VALID = 0
ALL_VALID = 1


class Tile(object):
  '''
  A tile of an array: an extent (offset + size) and data for that extent.
  '''

  @property
  def data(self):
    return self._data

  @data.setter
  def data(self, val):
    if val is not None:
      Assert.eq(val.dtype, self.dtype)
    self._data = val

  # BLOB INTERFACE
  def get(self, selector):
    self._initialize()
    try:
      return self.data[selector]
    except:
      util.log_info('SELECT FAILED: %s, %s', self._data.shape, selector)
      raise

  def update(self, update, reducer):
    return merge_tiles(self, update, reducer)

  def _initialize(self):
    # check if this is a scalar...
    if len(self.shape) == 0:
      return

    if self.data is None:
      self.data = np.ndarray(self.shape, dtype=self.dtype)

    # sparse matrix
    if not isinstance(self.data, np.ndarray):
      return

    if self.valid is NONE_VALID:
      self.valid = np.zeros(self.shape, dtype=np.bool)
    elif self.valid is ALL_VALID:
      self.valid = np.ones(self.shape, dtype=np.bool)
    else:
      assert isinstance(self.valid, np.ndarray)

  def __getitem__(self, idx):
    self._initialize()

    if len(self.shape) == 0:
      return self.data

    #if not np.all(self.valid[idx]):
    #  util.log_info('%s %s %s', idx, self.data[idx], self.valid[idx])
    #  raise ValueError

    return self.data[idx]

  def __setitem__(self, idx, val):
    self._initialize()
    self.valid[idx] = 1
    self.data[idx] = val

  def __repr__(self):
    return 'tile(%s, %s)' % (self.shape, self.dtype)


def from_data(data):
  t = Tile()
  t.dtype = data.dtype
  t.data = data
  t.valid = ALL_VALID
  t.shape = data.shape
  return t


def from_shape(shape, dtype):
  t = Tile()
  t.dtype = dtype
  t.shape = shape
  t.data = None
  t.valid = NONE_VALID
  return t


def from_intersection(src, overlap, data):
  '''
  Return a tile for ``src``, masked to update the area specifed by ``overlap``.
  
  :param src: `TileExtent`
  :param overlap: `TileExtent`
  :param data:
  '''
  t = Tile()
  t.dtype = data.dtype
  t.data = np.ndarray(src.shape, dtype=data.dtype)
  t.valid = np.zeros(src.shape, dtype=np.bool)
  t.shape = src.shape

  slc = extent.offset_slice(src, overlap)
  t.data[slc] = data
  t.valid[slc] = 1
  return t


def merge_tiles(old_tile, new_tile, reducer):
  Assert.isinstance(old_tile, Tile)
  Assert.isinstance(new_tile, Tile)

  Assert.eq(old_tile.dtype, new_tile.dtype)
  Assert.eq(old_tile.shape, new_tile.shape)

  old_tile._initialize()
  new_tile._initialize()

  #util.log_info('OLD: %s', old_tile.data)
  #util.log_info('NEW: %s', new_tile.data)

  # zero-dimensional arrays; just use
  # data == None as a mask.
  if len(old_tile.shape) == 0:
    if old_tile.data is None:
      old_tile.data = new_tile.data
    else:
      old_tile.data = reducer(old_tile.data, new_tile.data)
    return old_tile

  Assert.eq(old_tile.shape, new_tile.shape)
  replaced = ~old_tile.valid & new_tile.valid
  updated = old_tile.valid & new_tile.valid

  old_tile.data[replaced] = new_tile.data[replaced]
  old_tile.valid[replaced] = 1

  if np.any(updated):
    old_tile.data[updated] = reducer(old_tile.data[updated], new_tile.data[updated])

  return old_tile
