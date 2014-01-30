import unittest

import numpy as np

from spartan import expr
from spartan.util import Assert
from spartan import util
import test_common


TEST_SIZE = 50

class TestReduce(test_common.ClusterTest):
  TILE_SIZE = TEST_SIZE / 10
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
      Assert.all_eq(val, nx.argmax(axis))

  def test_simple_sum(self):
    def _(axis):
      util.log_info('Testing sum over axis %s', axis)
      a = expr.ones((TEST_SIZE, TEST_SIZE)) + expr.ones((TEST_SIZE, TEST_SIZE))
      b = a.sum(axis=axis)
      Assert.all_eq(b.glom(), 2 * np.ones((TEST_SIZE, TEST_SIZE)).sum(axis))

    _(axis=0)
    _(axis=1)
    _(axis=None)
    
  def test_count_nonzero(self):
    x = expr.ones((TEST_SIZE,))
    Assert.eq(expr.count_nonzero(x).glom(), TEST_SIZE)
    x = expr.zeros((TEST_SIZE,))
    Assert.eq(expr.count_nonzero(x).glom(), 0)

  def test_count_zero(self):
    x = expr.ones((TEST_SIZE,))
    Assert.eq(expr.count_zero(x).glom(), 0)
    x = expr.zeros((TEST_SIZE,))
    Assert.eq(expr.count_zero(x).glom(), TEST_SIZE)

if __name__ == '__main__':
#   x = TestReduce(methodName='test_simple_sum')
#   x.setUpClass()
#   for i in range(100):
#     x.setUp()
#     x.test_simple_sum()
  unittest.main()
