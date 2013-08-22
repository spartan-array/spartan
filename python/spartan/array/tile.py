from spartan import util
from spartan.util import Assert
import numpy as np


class Tile(object):
  '''
  A tile of an array: an extent (offset + size) and data for that extent.
  '''
  def __init__(self, extent, data=None, dtype=None):
    self.extent = extent
    
    if data is not None:
      self.data = data
      self.mask = np.zeros(self.data.shape, dtype=np.bool)
      self.dtype = data.dtype
      Assert.eq(extent.shape, data.shape)
      assert dtype is None or dtype == data.dtype, 'Datatype mismatch.'
    else:
      # mask and data will be initialized on demand.
      self.mask = None
      self.data = None
      assert dtype is not None
      self.dtype = dtype
    
  def _initialize(self):
    if self.extent.ndim == 0:
      return
  
    if self.mask is None:
      self.mask = np.ones(self.extent.shape, dtype=np.bool)
      
    if self.data is None:
      self.data = np.ndarray(self.extent.shape, dtype=self.dtype)

  def get(self, slc):
    self._initialize()
    
  @property
  def shape(self):
    return self.extent.shape
    
  def __getitem__(self, idx):
    self._initialize()
    if self.extent.ndim == 0:
      return self.data
    
    assert np.all(~self.mask[idx])
    return self.data[idx] 
  
  def __setitem__(self, idx, val):
    self._initialize()

    data = self.data[idx]
    invalid = self.mask[idx]
    self.mask[invalid] = False
    data[idx] = val


def make_tile(extent, data):
    return Tile(extent, data=data)


class TileAccum(object):
  def __init__(self, accum):
    self.accum = accum
  
  def __call__(self, old_tile, new_tile):
    Assert.is_instance(old_tile, Tile)
    Assert.is_instance(new_tile, Tile)
    
    old_tile._initialize()
 
    # zero-dimensional arrays; just use 
    # data == None as a mask. 
    if old_tile.extent.ndim == 0:
      if old_tile.data is None:
        old_tile.data = new_tile.data
      else:
        old_tile.data = self.accum(old_tile.data, new_tile.data)
      return old_tile
    
    idx = old_tile.extent.local_offset(new_tile.extent)
    data = old_tile.data[idx]
    
    invalid = old_tile.mask[idx]
    valid = ~old_tile.mask[idx]
    
    data[invalid] = new_tile.data[invalid]
    if data[valid].size > 0:
      data[valid] = self.accum(data[valid], new_tile.data[valid])
    old_tile.mask[idx] = False
#     util.log('%s, %s', old_tile.mask, idx)
    return old_tile
