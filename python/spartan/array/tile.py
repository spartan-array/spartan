from spartan import util
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
      assert dtype is None or dtype == data.dtype, 'Datatype mismatch.'
    else:
      # mask and data will be initialized on demand.
      self.mask = None
      self.data = None
      assert dtype is not None
      self.dtype = dtype
    
  def _initialize(self):
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
