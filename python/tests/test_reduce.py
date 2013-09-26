from spartan.array import expr, distarray
from spartan.util import Assert
import numpy as np
import test_common

TEST_SIZE = 20

def test_reduce_3d(ctx):
  distarray.TILE_SIZE = TEST_SIZE ** 3 / 16
  x = expr.arange((TEST_SIZE, TEST_SIZE, TEST_SIZE), dtype=np.int64)
  nx = np.arange(TEST_SIZE * TEST_SIZE * TEST_SIZE, dtype=np.int64).reshape((TEST_SIZE, TEST_SIZE, TEST_SIZE))
  
  Assert.all_eq(nx, x.evaluate().glom())
  y = x.sum()
  val = y.evaluate().glom()
  Assert.all_eq(val, nx.sum())

def test_reduce_2d(ctx):
  distarray.TILE_SIZE = TEST_SIZE ** 2 / 16
  x = expr.arange((TEST_SIZE, TEST_SIZE), dtype=np.int)
  nx = np.arange(TEST_SIZE * TEST_SIZE, dtype=np.int).reshape((TEST_SIZE, TEST_SIZE))
  y = x.sum()
  val = y.evaluate().glom()
  Assert.all_eq(val, nx.sum())
  
def test_reduce_1d(ctx):
  distarray.TILE_SIZE = TEST_SIZE / 2
  x = expr.arange((TEST_SIZE,), dtype=np.int)
  nx = np.arange(TEST_SIZE, dtype=np.int)
  y = x.sum()
  val = y.evaluate().glom()
  Assert.all_eq(val, nx.sum())
  
if __name__ == '__main__':
  test_common.run_cluster_tests(__file__)