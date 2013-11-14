import numpy as np
from scipy.sparse import dok_matrix
from . import extent
from spartan import util
from spartan.util import Assert


NONE_VALID = 0
ALL_VALID = 1

TYPE_DENSE = 0
TYPE_SPARSE = 1

class Tile(object):
  '''
  A tile of an array: an extent (offset + size) and data for that extent.

  Tiles implement the Blob protocol: see `Blob`.
  '''
  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = dtype

  @property
  def data(self):
    return self._data

  @data.setter
  def data(self, val):
    if val is not None:
      Assert.eq(val.dtype, self.dtype)
    self._data = val

  def get(self, selector):
    self._initialize()
    if extent.is_complete(self.data.shape, selector):
      return self.data
    return self.data[selector]

  def __getitem__(self, idx):
    self._initialize()

    if len(self.shape) == 0:
      return self.data

    #if not np.all(self.mask[idx]):
    #  util.log_info('%s %s %s', idx, self.data[idx], self.mask[idx])
    #  raise ValueError
    return self.data[idx]


class DenseTile(Tile):
  def __init__(self, shape, dtype, data, mask=None):
    Tile.__init__(self, shape, dtype)
    if data is not None:
      Assert.eq(data.shape, shape)

    self._data = data

    if mask is None:
      if data is None:
        self.mask = NONE_VALID
      else:
        self.mask = ALL_VALID
    else:
      Assert.eq(mask.shape, shape)
      Assert.eq(mask.dtype, np.bool)
      self.mask = mask

    self.shape = shape
    self.dtype = dtype

  def update(self, update, reducer):
    return merge_dense(self, update, reducer)

  def _initialize(self):
    # if this is a scalar, then data is just the scalar value itself.
    if len(self.shape) == 0:
      return

    if self.data is None:
      self._data = np.ndarray(self.shape, dtype=self.dtype)

    Assert.isinstance(self.data, np.ndarray)

    if self.mask is NONE_VALID:
      self.mask = np.zeros(self.shape, dtype=np.bool)
    elif self.mask is ALL_VALID:
      self.mask = np.ones(self.shape, dtype=np.bool)
    else:
      assert isinstance(self.mask, np.ndarray)

  def __setitem__(self, idx, val):
    self._initialize()

    if self._type == TYPE_SPARSE:
      self.data[idx] = val
    else:
      self.mask[idx] = 1
      self.data[idx] = val

  def __repr__(self):
    return 'dense(%s, %s)' % (self.shape, self.dtype)

class SparseTile(Tile):
  def __init__(self, shape, data, dtype):
    Tile.__init__(self, shape, dtype)
    self._data = data

  def _initialize(self):
    if self.data is None:
      self._data = dok_matrix(self.shape, self.dtype)

def from_data(data):
  if isinstance(data, np.ndarray) or np.isscalar(data):
    return DenseTile(
                shape=data.shape,
                data=data,
                dtype=data.dtype)
  else:
    return SparseTile(
                shape=data.shape,
                data=data,
                dtype=data.dtype)


def from_shape(shape, dtype, tile_type=TYPE_DENSE):
  assert tile_type == TYPE_DENSE
  return DenseTile(shape =shape, data = None, dtype = dtype)


def from_intersection(src, overlap, data):
  '''
  Return a tile for ``src``, masked to update the area specifed by ``overlap``.
  
  :param src: `TileExtent`
  :param overlap: `TileExtent`
  :param data:
  '''
  slc = extent.offset_slice(src, overlap)
  tdata = np.ndarray(src.shape, dtype=data.dtype)
  tmask = np.zeros(src.shape, dtype=np.bool)
  tdata[slc] = data
  tmask[slc] = 1

  return DenseTile(dtype=data.dtype, data=tdata, mask=tmask, shape=src.shape)


def merge_dense(old_tile, new_tile, reducer):
  Assert.isinstance(old_tile, Tile)
  Assert.isinstance(new_tile, Tile)

  Assert.eq(old_tile.dtype, new_tile.dtype)
  Assert.eq(old_tile.shape, new_tile.shape)

  if old_tile.data is None:
    return new_tile

  old_tile._initialize()
  new_tile._initialize()

  #util.log_info('OLD: %s', old_tile.data)
  #util.log_info('NEW: %s', new_tile.data)

  # zero-dimensional arrays; just use
  # data == None as a mask.
  if len(old_tile.shape) == 0:
    if old_tile.data is None or reducer is None:
      old_tile.data = new_tile.data
    else:
      old_tile.data = reducer(old_tile.data, new_tile.data)
    return old_tile

  Assert.eq(old_tile.shape, new_tile.shape)
  replaced = ~old_tile.mask & new_tile.mask
  updated = old_tile.mask & new_tile.mask

  old_tile.data[replaced] = new_tile.data[replaced]
  old_tile.mask[replaced] = 1

  if np.any(updated):
    if reducer is None:
      old_tile.data[updated] = new_tile.data[updated]
    else:
      old_tile.data[updated] = reducer(old_tile.data[updated], new_tile.data[updated])

  return old_tile
