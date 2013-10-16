from spartan import expr
from spartan.dense import distarray
from spartan.util import Assert
from test_common import with_ctx
import numpy as np
import spartan
import test_common

TEST_SIZE = 20

@with_ctx
def test_sum_3d(ctx):
  distarray.TILE_SIZE = TEST_SIZE ** 3 / 16
  x = expr.arange((TEST_SIZE, TEST_SIZE, TEST_SIZE), dtype=np.int64)
  nx = np.arange(TEST_SIZE * TEST_SIZE * TEST_SIZE, dtype=np.int64).reshape((TEST_SIZE, TEST_SIZE, TEST_SIZE))

  for axis in [None, 0, 1, 2]:  
    y = x.sum(axis)
    val = y.glom()
    Assert.all_eq(val, nx.sum(axis))

@with_ctx
def test_sum_2d(ctx):
  distarray.TILE_SIZE = TEST_SIZE ** 2 / 16
  x = expr.arange((TEST_SIZE, TEST_SIZE), dtype=np.int)
  nx = np.arange(TEST_SIZE * TEST_SIZE, dtype=np.int).reshape((TEST_SIZE, TEST_SIZE))
  for axis in [None, 0, 1]:  
    y = x.sum(axis)
    val = y.glom()
    Assert.all_eq(val, nx.sum(axis))
  
@with_ctx
def test_sum_1d(ctx):
  distarray.TILE_SIZE = TEST_SIZE / 2
  x = expr.arange((TEST_SIZE,), dtype=np.int)
  nx = np.arange(TEST_SIZE, dtype=np.int)
  y = x.sum()
  val = y.glom()
  Assert.all_eq(val, nx.sum())
  
@with_ctx
def test_argmin_1d(ctx):
  distarray.TILE_SIZE = TEST_SIZE / 2
  x = expr.arange((TEST_SIZE,), dtype=np.int)
  nx = np.arange(TEST_SIZE, dtype=np.int)
  y = x.argmin()
  val = y.glom()
  Assert.all_eq(val, nx.argmin())

@with_ctx
def test_argmin_2d(ctx):
  distarray.TILE_SIZE = TEST_SIZE ** 2 / 16
  
  for axis in [1]: #[None, 0, 1]:
    x = expr.arange((TEST_SIZE, TEST_SIZE), dtype=np.int)
    nx = np.arange(TEST_SIZE * TEST_SIZE, dtype=np.int).reshape((TEST_SIZE, TEST_SIZE))
    y = x.argmin(axis=axis)
    val = expr.glom(y)
    Assert.all_eq(val, nx.argmin(axis=axis))
  
@with_ctx
def test_argmin_3d(ctx):
  distarray.TILE_SIZE = TEST_SIZE ** 3 / 16
  x = expr.arange((TEST_SIZE, TEST_SIZE, TEST_SIZE), dtype=np.int64)
  nx = np.arange(TEST_SIZE * TEST_SIZE * TEST_SIZE, dtype=np.int64).reshape((TEST_SIZE, TEST_SIZE, TEST_SIZE))

  for axis in [None, 0, 1, 2]:  
    y = x.argmin(axis)
    val = y.glom()
    Assert.all_eq(val, nx.argmin(axis))
