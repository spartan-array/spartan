from spartan.array import expr, distarray
from spartan.util import Assert
import numpy as np
import test_common

TEST_SIZE = 20


def test_sum_3d(ctx):
  distarray.TILE_SIZE = TEST_SIZE ** 3 / 16
  x = expr.arange((TEST_SIZE, TEST_SIZE, TEST_SIZE), dtype=np.int64)
  nx = np.arange(TEST_SIZE * TEST_SIZE * TEST_SIZE, dtype=np.int64).reshape((TEST_SIZE, TEST_SIZE, TEST_SIZE))

  for axis in [None, 0, 1, 2]:  
    y = x.sum(axis)
    val = y.evaluate().glom()
    Assert.all_eq(val, nx.sum(axis))

def test_sum_2d(ctx):
  distarray.TILE_SIZE = TEST_SIZE ** 2 / 16
  x = expr.arange((TEST_SIZE, TEST_SIZE), dtype=np.int)
  nx = np.arange(TEST_SIZE * TEST_SIZE, dtype=np.int).reshape((TEST_SIZE, TEST_SIZE))
  for axis in [None, 0, 1]:  
    y = x.sum(axis)
    val = y.evaluate().glom()
    Assert.all_eq(val, nx.sum(axis))
  
def test_sum_1d(ctx):
  distarray.TILE_SIZE = TEST_SIZE / 2
  x = expr.arange((TEST_SIZE,), dtype=np.int)
  nx = np.arange(TEST_SIZE, dtype=np.int)
  y = x.sum()
  val = y.evaluate().glom()
  Assert.all_eq(val, nx.sum())
  
def test_argmin_1d(ctx):
  distarray.TILE_SIZE = TEST_SIZE / 2
  x = expr.arange((TEST_SIZE,), dtype=np.int)
  nx = np.arange(TEST_SIZE, dtype=np.int)
  y = x.argmin()
  val = y.evaluate().glom()
  Assert.all_eq(val, nx.argmin())

def test_argmin_2d(ctx):
  distarray.TILE_SIZE = TEST_SIZE ** 2 / 16
  x = expr.arange((TEST_SIZE, TEST_SIZE), dtype=np.int)
  nx = np.arange(TEST_SIZE * TEST_SIZE, dtype=np.int).reshape((TEST_SIZE, TEST_SIZE))
  
  for axis in [None, 0, 1]:
    y = x.argmin(axis=axis)
    val = y.evaluate().glom()
    Assert.all_eq(val, nx.argmin(axis=axis))
  
def test_argmin_3d(ctx):
  distarray.TILE_SIZE = TEST_SIZE ** 3 / 16
  x = expr.arange((TEST_SIZE, TEST_SIZE, TEST_SIZE), dtype=np.int64)
  nx = np.arange(TEST_SIZE * TEST_SIZE * TEST_SIZE, dtype=np.int64).reshape((TEST_SIZE, TEST_SIZE, TEST_SIZE))

  for axis in [None, 0, 1, 2]:  
    y = x.argmin(axis)
    val = y.evaluate().glom()
    Assert.all_eq(val, nx.argmin(axis))
  
if __name__ == '__main__':
  test_common.run(__file__)