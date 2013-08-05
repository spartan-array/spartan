from .. import util
import numpy as np


class Tile(object):
  '''
  A tile of an array: an extent (offset + size) and data for that extent.
  '''
  def __init__(self, extent, data, dtype):
    self.extent = extent
    self.data = data
    self.dtype = dtype

  def update(self, tile, accumulator):
    idx = self.extent.local_offset(tile.extent)
    self.data[idx] = accumulator(self.data[idx], tile.data)
    

class MaskedTile(Tile):
  def __init__(self, extent, data, dtype):
    Tile.__init__(self, extent, data, dtype)
    
    # mask will be initialized when data is first requested
    # or updated
    self.mask = None
    
  def _initialize(self):
    if self.mask is None:
      self.mask = np.ones(self.data.shape, dtype=np.bool)
      
    if self.data is None:
      self.data = np.ndarray(self.data.shape, dtype=self.dtype)

  def get(self, slc):
    self._initialize()
    pass
    

def make_tile(extent, data, dtype, masked):
  if masked:
    return MaskedTile(extent, data, dtype)
  else:
    return Tile(extent, data, dtype)
