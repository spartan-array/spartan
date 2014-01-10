import unittest

import numpy as np
from spartan import expr, util
from spartan.array import tile
from spartan.util import Assert
import test_common


ARRAY_SIZE = (10, 10)

class TestReduce(test_common.ClusterTest):
  def test_create_dense(self):
    t = tile.from_shape((10, 10), dtype=np.float32, tile_type=tile.TYPE_DENSE)
    t._initialize()
    Assert.eq(t.data.shape, (10, 10))
    Assert.eq(t.mask.shape, (10, 10))
    
  def test_create_sparse(self):
    t = tile.from_shape((10, 10), dtype=np.float32, tile_type=tile.TYPE_SPARSE)
    t._initialize()
    Assert.eq(t.data.shape, (10, 10))
    Assert.eq(t.mask, None) 

if __name__ == '__main__':
  unittest.main()
