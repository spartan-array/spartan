import unittest

import numpy as np
from spartan import expr, util
from spartan.array import tile
from spartan.util import Assert
import spartan.array.extent as extent
import test_common
import scipy.sparse as sp


ARRAY_SIZE = (10, 10)
UPDATE_SHAPE = (8, 8)
UPDATE_SUBSLICE = extent.create((0,0),(8,8), UPDATE_SHAPE).to_slice()

class TestTile(test_common.ClusterTest):
  def test_create_dense(self):
    t = tile.from_shape(ARRAY_SIZE, dtype=np.float64, tile_type=tile.TYPE_DENSE)
    t._initialize()
    Assert.eq(t.mask.shape, ARRAY_SIZE)
    
  def test_create_sparse(self):
    t = tile.from_shape(ARRAY_SIZE, dtype=np.float64, tile_type=tile.TYPE_SPARSE)
    t._initialize()
    Assert.eq(t.data.shape, ARRAY_SIZE)
    Assert.eq(t.mask, None) 

  def test_update_dense_to_dense(self):
    t = tile.from_shape(ARRAY_SIZE, dtype=np.float64, tile_type=tile.TYPE_DENSE)
    update = np.ones(UPDATE_SHAPE)
    t.update(UPDATE_SUBSLICE, update, None)
    print t.data
    
  def test_update_dense_to_sparse(self):
    t = tile.from_shape(ARRAY_SIZE, dtype=np.float64, tile_type=tile.TYPE_SPARSE)
    update = np.ones(UPDATE_SHAPE)
    t.update(UPDATE_SUBSLICE, update, None)
    print t.data.todense()
    
  def test_update_sparse_to_dense(self):
    t = tile.from_shape(ARRAY_SIZE, dtype=np.float64, tile_type=tile.TYPE_DENSE)
    update = sp.lil_matrix(ARRAY_SIZE)
    for i in range(UPDATE_SHAPE[0]):
      update[i,i] = 1
    t.update(UPDATE_SUBSLICE, update, None)
    print t.data
    print t.mask
    
  def test_update_sparse_to_sparse(self):
    t = tile.from_shape(ARRAY_SIZE, dtype=np.float64, tile_type=tile.TYPE_SPARSE)
    update = sp.lil_matrix(ARRAY_SIZE)
    for i in range(UPDATE_SHAPE[0]):
      update[i,i] = 1
    t.update(UPDATE_SUBSLICE, update, None)
    Assert.eq(sp.issparse(t.data), True)
    print t.data.todense()
    
if __name__ == '__main__':
  unittest.main()
