from spartan import expr
from spartan.array import distarray
from spartan.util import Assert
import numpy as np
import spartan
import test_common

TEST_SIZE = 5

class TestReduce(test_common.ClusterTest):
  TILE_SIZE = TEST_SIZE
  def test_sum_3d(self):
    x = expr.arange((TEST_SIZE, TEST_SIZE, TEST_SIZE), dtype=np.int64)
    nx = np.arange(TEST_SIZE * TEST_SIZE * TEST_SIZE, dtype=np.int64).reshape((TEST_SIZE, TEST_SIZE, TEST_SIZE))
  
    for axis in [None, 0, 1, 2]:  
      y = x.sum(axis)
      val = y.glom()
      Assert.all_eq(val, nx.sum(axis))
  
  def test_sum_2d(self):
    x = expr.arange((TEST_SIZE, TEST_SIZE), dtype=np.int)
    nx = np.arange(TEST_SIZE * TEST_SIZE, dtype=np.int).reshape((TEST_SIZE, TEST_SIZE))
    for axis in [None, 0, 1]:  
      y = x.sum(axis)
      val = y.glom()
      Assert.all_eq(val, nx.sum(axis))
    
  def test_sum_1d(self):
    x = expr.arange((TEST_SIZE,), dtype=np.int)
    nx = np.arange(TEST_SIZE, dtype=np.int)
    y = x.sum()
    val = y.glom()
    Assert.all_eq(val, nx.sum())
    
  def test_argmin_1d(self):
    x = expr.arange((TEST_SIZE,), dtype=np.int)
    nx = np.arange(TEST_SIZE, dtype=np.int)
    y = x.argmin()
    val = y.glom()
    Assert.all_eq(val, nx.argmin())
  
  def test_argmin_2d(self):
    for axis in [1]: #[None, 0, 1]:
      x = expr.arange((TEST_SIZE, TEST_SIZE), dtype=np.int)
      nx = np.arange(TEST_SIZE * TEST_SIZE, dtype=np.int).reshape((TEST_SIZE, TEST_SIZE))
      y = x.argmin(axis=axis)
      val = expr.glom(y)
      Assert.all_eq(val, nx.argmin(axis=axis))
    
  def test_argmin_3d(self):
    x = expr.arange((TEST_SIZE, TEST_SIZE, TEST_SIZE), dtype=np.int64)
    nx = np.arange(TEST_SIZE * TEST_SIZE * TEST_SIZE, dtype=np.int64).reshape((TEST_SIZE, TEST_SIZE, TEST_SIZE))
  
    for axis in [None, 0, 1, 2]:  
      y = x.argmin(axis)
      val = y.glom()
      Assert.all_eq(val, nx.argmin(axis))

  def test_argmax_1d(self):
    x = expr.arange((TEST_SIZE,), dtype=np.int)
    nx = np.arange(TEST_SIZE, dtype=np.int)
    y = x.argmax()
    val = y.glom()
    Assert.all_eq(val, nx.argmax())
  
  def test_argmax_2d(self):
    for axis in [1]: #[None, 0, 1]:
      x = expr.arange((TEST_SIZE, TEST_SIZE), dtype=np.int)
      nx = np.arange(TEST_SIZE * TEST_SIZE, dtype=np.int).reshape((TEST_SIZE, TEST_SIZE))
      y = x.argmax(axis=axis)
      val = expr.glom(y)
      Assert.all_eq(val, nx.argmax(axis=axis))
    
  def test_argmax_3d(self):
    x = expr.arange((TEST_SIZE, TEST_SIZE, TEST_SIZE), dtype=np.int64)
    nx = np.arange(TEST_SIZE * TEST_SIZE * TEST_SIZE, dtype=np.int64).reshape((TEST_SIZE, TEST_SIZE, TEST_SIZE))
  
    for axis in [None, 0, 1, 2]:  
      y = x.argmax(axis)
      val = y.glom()
      print val
      Assert.all_eq(val, nx.argmax(axis))