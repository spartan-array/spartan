import numpy as np
from scipy.sparse import dok_matrix
from . import extent
from spartan import util
from spartan.util import Assert


MASK_VALID = 1
MASK_INVALID = 0

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
    raise NotImplementedError

  def update(self, update, reducer):
    raise NotImplementedError


class DenseTile(Tile):
  '''Dense tiles use regular numpy arrays for storage.

  They track updates using a mask to determine valid regions.
  This masking is done manually, instead of using numpy.ma (MaskedArray),
  as MaskedArrays do not seem to properly handle masking of
  structured arrays.  Conversion to MaskedArrays is performed on
  fetch.
  '''

  def __init__(self, shape, dtype, data, mask=None):
    Tile.__init__(self, shape, dtype)

    if data is not None:
      Assert.eq(data.shape, shape)
      if mask is None:
        mask = MASK_VALID
    else:
      mask = MASK_INVALID

    self.mask = mask
    self.data = data
    self.shape = shape
    self.dtype = dtype

  def update(self, update, reducer):
    return merge_dense(self, update, reducer)

  def get(self, selector=None):
    self._initialize()

    # scalars are just returned directly.
    if len(self.data.shape) == 0:
      return self.data

    if selector is None or extent.is_complete(self.data.shape, selector):
      data = self.data
      mask = self.mask
    else:
      Assert.le(len(selector), len(self.data.shape),
                'Selector has more dimensions than data! %s %s' % (selector, self.data.shape))

      data = self.data[selector]
      mask = self.mask[selector]

    masked = np.ma.masked_all(data.shape, dtype=data.dtype)
    #util.log_info("\nM:%s\nD%s\n", data, mask)
    masked[mask] = data[mask]
    return masked

  def _initialize(self):
    # if this is a scalar, then data is just the scalar value itself.
    if len(self.shape) == 0:
      return

    if self.data is None:
      self.data = np.ndarray(self.shape, dtype=self.dtype)

    if not isinstance(self.mask, np.ndarray):
      mask_val = self.mask
      self.mask = np.ndarray(self.shape, dtype=np.bool)
      self.mask[:] = mask_val

  def __setitem__(self, idx, val):
    self._initialize()
    self.mask[idx] = MASK_VALID
    self.data[idx] = val

  def __repr__(self):
    return 'dense(%s, %s)' % (self.shape, self.dtype)


class SparseTile(Tile):
  def __init__(self, shape, data, dtype):
    Tile.__init__(self, shape, dtype)
    self.data = data

  def get(self, selector=None):
    self._initialize()
    assert selector is None or extent.is_complete(self.shape, selector)
    return self.data

  def _initialize(self):
    if self.data is None:
      self.data = dok_matrix(self.shape, self.dtype)

  def __setitem__(self, idx, val):
    self._initialize()
    self.data[idx] = val


def from_data(data):
  if isinstance(data, np.ndarray) or np.isscalar(data):
    return DenseTile(
      shape=data.shape,
      data=data,
      dtype=data.dtype)
  else:
    return SparseTile(shape=data.shape, data=data, dtype=data.dtype)


def from_shape(shape, dtype, tile_type=TYPE_DENSE):
  assert tile_type == TYPE_DENSE
  return DenseTile(shape=shape, data=None, dtype=dtype)


def from_intersection(src, overlap, data):
  '''
  Return a tile for ``src``, masked to update the area specifed by ``overlap``.
  
  :param src: `TileExtent`
  :param overlap: `TileExtent`
  :param data:
  '''
  slc = extent.offset_slice(src, overlap)
  tdata = np.ndarray(src.shape, data.dtype)
  tmask = np.ndarray(src.shape, np.bool)
  tmask[:] = MASK_INVALID
  tmask[slc] = MASK_VALID
  tdata[slc] = data
  return DenseTile(dtype=data.dtype, data=tdata, shape=src.shape, mask=tmask)


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
  old_tile.mask[replaced] = MASK_VALID

  if np.any(updated):
    if reducer is None:
      old_tile.data[updated] = new_tile.data[updated]
    else:
      old_tile.data[updated] = reducer(old_tile.data[updated], new_tile.data[updated])

  return old_tile
