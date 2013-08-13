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

    # mask will be initialized when data is first requested
    # or updated
    self.mask = None
    
  def _initialize(self):
    if self.mask is None:
      self.mask = np.ones(self.extent.shape, dtype=np.bool)
      
    if self.data is None:
      self.data = np.ndarray(self.extent.shape, dtype=self.dtype)

  def get(self, slc):
    self._initialize()
    
  def __getitem__(self, idx):
    self._initialize()
    assert np.all(~self.mask[idx])
    
    return self.data[idx] 
  
  
  def __setitem__(self, idx, val):
    self._initialize()

    data = self.data[idx]
    invalid = self.mask[idx]
    self.mask[invalid] = False
    data[idx] = val


def make_tile(extent, data, dtype, masked):
    return Tile(extent, data, dtype)
