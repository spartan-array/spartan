from spartan.array import expr, distarray
from spartan.util import Assert
import numpy as np
import test_common

DIM = 20

def test_reduce_3d(ctx):
  distarray.TILE_SIZE = DIM ** 3 / 16
  x = expr.arange((DIM, DIM, DIM), dtype=np.int64)
  nx = np.arange(DIM * DIM * DIM, dtype=np.int64).reshape((DIM, DIM, DIM))
  
  Assert.all_eq(nx, x.evaluate().glom())
  y = x.sum()
  val = y.evaluate().glom()
  Assert.all_eq(val, nx.sum())

def test_reduce_2d(ctx):
  distarray.TILE_SIZE = DIM ** 2 / 16
  x = expr.arange((DIM, DIM), dtype=np.int)
  nx = np.arange(DIM * DIM, dtype=np.int).reshape((DIM, DIM))
  y = x.sum()
  val = y.evaluate().glom()
  Assert.all_eq(val, nx.sum())
  
def test_reduce_1d(ctx):
  distarray.TILE_SIZE = DIM / 2
  x = expr.arange((DIM,), dtype=np.int)
  nx = np.arange(DIM, dtype=np.int)
  y = x.sum()
  val = y.evaluate().glom()
  Assert.all_eq(val, nx.sum())
  
if __name__ == '__main__':
  test_common.run_cluster_tests(__file__)