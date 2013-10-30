import math
import sys
import unittest

import spartan
import numpy as np
from spartan import expr, util
from spartan.dense import distarray, extent
from spartan.util import Assert
import test_common


TEST_SIZE = 100

def add_one_extent(v, ex):
  util.log_info('Mapping: %s', ex)
  yield (ex, v.fetch(ex) + 1)

def add_one_tile(tiles):
  return tiles[0] + 1

class SliceTest(test_common.ClusterTest):
  def test_slice_get(self):
    x = expr.arange((TEST_SIZE, TEST_SIZE))
    z = x[5:8, 5:8]
    zc = expr.dag(z)
    val = expr.force(zc)
    nx = np.arange(TEST_SIZE*TEST_SIZE).reshape(TEST_SIZE, TEST_SIZE)
    Assert.all_eq(val.glom(), nx[5:8, 5:8])
  
  def test_slice_map(self):
    x = expr.arange((TEST_SIZE, TEST_SIZE))
    z = x[5:8, 5:8]
    z = expr.map(z, add_one_tile) 
    val = expr.force(z)
    nx = np.arange(TEST_SIZE*TEST_SIZE).reshape(TEST_SIZE, TEST_SIZE)
    
    Assert.all_eq(val.glom(), nx[5:8, 5:8] + 1)
  
  
  def test_slice_shuffle(self):
    x = expr.arange((TEST_SIZE, TEST_SIZE))
    z = x[5:8, 5:8]
    z = expr.shuffle(z, add_one_extent) 
    val = z.force()
    nx = np.arange(TEST_SIZE*TEST_SIZE).reshape(TEST_SIZE, TEST_SIZE)
    
    Assert.all_eq(val.glom(), nx[5:8, 5:8] + 1)
    
  def test_slice_map2(self):
    x = expr.arange((10, 10, 10), dtype=np.int)
    nx = np.arange(10 * 10 * 10, dtype=np.int).reshape((10, 10, 10))
    
    y = x[:, :, 0]
    z = expr.map(y, lambda tiles: tiles[0] + 13)
    val = z.glom()
   
    Assert.all_eq(val.reshape(10, 10), nx[:, :, 0] + 13)
    
  def test_from_slice(self):
    print extent.from_slice((slice(None), slice(None), 0), [100, 100, 100])
  
  def test_slice_reduce(self):
    x = expr.arange((TEST_SIZE, TEST_SIZE, TEST_SIZE), dtype=np.int)
    nx = np.arange(TEST_SIZE * TEST_SIZE * TEST_SIZE, dtype=np.int).reshape((TEST_SIZE, TEST_SIZE, TEST_SIZE))
    y = x[:, :, 0].sum()
    val = y.glom()
    
    Assert.all_eq(val, nx[:, :, 0].sum())

if __name__ == '__main__':
  rest = spartan.config.parse_args(sys.argv)
  unittest.main(argv=rest) 