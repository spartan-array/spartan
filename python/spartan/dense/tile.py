from . import extent
from spartan import util
from spartan.util import Assert
import numpy as np


class Tile(object):
  '''
  A tile of an array: an extent (offset + size) and data for that extent.
  '''
    
  def _initialize(self):
    # check if this is a scalar...
    if len(self.shape) == 0:
      return
  
    if self.valid is None:
      self.valid = np.zeros(self.shape, dtype=np.bool)
      
    if self.data is None:
      self.data = np.ndarray(self.shape, dtype=self.dtype)

  def get(self, slc):
    self._initialize()
    
  def __getitem__(self, idx):
    self._initialize()
    if len(self.shape) == 0:
      return self.data
    
    if not np.all(self.valid[idx]):
      util.log('%s %s %s', idx, self.data[idx], self.valid[idx])
      raise ValueError
    
    return self.data[idx] 
  
  def __setitem__(self, idx, val):
    self._initialize()
    self.valid[idx] = 1
    self.data[idx] = val
    
  def __repr__(self):
    return 'tile(%s, %s)' % (self.shape, self.dtype)

def from_data(data):
  t = Tile()
  t.data = data
  t.dtype = data.dtype
  t.valid = np.ones(data.shape, dtype=np.bool)
  t.shape = data.shape
  return t

def from_shape(shape, dtype):
  t = Tile()
  t.shape = shape
  t.data = None
  t.valid = None
  t.dtype = dtype
  return t

def from_intersection(src, overlap, data):
  '''
  Return a tile for ``src``, masked to update the area specifed by ``overlap``.
  
  :param src: `TileExtent`
  :param overlap: `TileExtent`
  :param data:
  '''
  t = Tile()
  t.data = np.ndarray(src.shape)
  t.valid = np.zeros(src.shape, dtype=np.bool)
  t.shape = src.shape
  t.dtype = data.dtype
  
  slc = extent.offset_slice(src, overlap)
  t.data[slc] = data
  t.valid[slc] = 1
  return t
  

class TileAccum(object):
  def __init__(self, accum):
    self.accum = accum
  
  def __call__(self, key, old_tile, new_tile):
    Assert.isinstance(old_tile, Tile)
    Assert.isinstance(new_tile, Tile)
    
    old_tile._initialize()
 
    # zero-dimensional arrays; just use 
    # data == None as a mask. 
    if len(old_tile.shape) == 0:
      if old_tile.data is None:
        old_tile.data = new_tile.data
      else:
        old_tile.data = self.accum(old_tile.data, new_tile.data)
      return old_tile
    
    Assert.eq(old_tile.shape, new_tile.shape)
    replaced = ~old_tile.valid & new_tile.valid    
    updated = old_tile.valid & new_tile.valid
    
    old_tile.data[replaced] = new_tile.data[replaced]
    old_tile.valid[replaced] = 1
    
    if np.any(updated):
      old_tile.data[updated] = self.accum(
                  old_tile.data[updated],
                  new_tile.data[updated])
    
    return old_tile
