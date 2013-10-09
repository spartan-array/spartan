from spartan import util
from spartan.array import prims, compile_expr, backend, expr
from spartan.dense import distarray, extent
from spartan.util import Assert
import math
import numpy as np
import test_common
from test_common import with_ctx

TEST_SIZE = 100
distarray.TILE_SIZE = TEST_SIZE ** 2 / 4

@with_ctx
def test_slice_get(ctx):
  x = expr.arange((TEST_SIZE, TEST_SIZE))
  z = x[5:8, 5:8]
  zc = compile_expr.compile(z)
  val = backend.evaluate(ctx, zc)
  nx = np.arange(TEST_SIZE*TEST_SIZE).reshape(TEST_SIZE, TEST_SIZE)
  Assert.all_eq(val.glom(), nx[5:8, 5:8])

def add_one_tile(tiles):
  return tiles[0] + 1

@with_ctx
def test_slice_map_tiles(ctx):
  x = expr.arange((TEST_SIZE, TEST_SIZE))
  z = x[5:8, 5:8]
  z = expr.map_tiles(z, add_one_tile) 
  zc = compile_expr.compile(z)
  val = backend.evaluate(ctx, zc)
  nx = np.arange(TEST_SIZE*TEST_SIZE).reshape(TEST_SIZE, TEST_SIZE)
  
  Assert.all_eq(val.glom(), nx[5:8, 5:8] + 1)

def add_one_extent(inputs, ex):
  util.log('Mapping: %s', ex)
  return (ex, inputs[0].fetch(ex) + 1)

@with_ctx
def test_slice_map_extents(ctx):
  x = expr.arange((TEST_SIZE, TEST_SIZE))
  z = x[5:8, 5:8]
  z = expr.map_extents(z, add_one_extent) 
  zc = compile_expr.compile(z)
  val = backend.evaluate(ctx, zc)
  nx = np.arange(TEST_SIZE*TEST_SIZE).reshape(TEST_SIZE, TEST_SIZE)
  
  Assert.all_eq(val.glom(), nx[5:8, 5:8] + 1)
  
  
@with_ctx
def test_slice_map_tiles2(ctx):
  x = expr.arange((10, 10, 10), dtype=np.int)
  nx = np.arange(10 * 10 * 10, dtype=np.int).reshape((10, 10, 10))
  
  y = x[:, :, 0]
  z = expr.map_tiles(y, lambda tiles: tiles[0] + 13)
  val = z.evaluate().glom()
  
  Assert.all_eq(val.reshape(10, 10), nx[:, :, 0] + 13)
  
@with_ctx
def test_from_slice(ctx):
  print extent.from_slice((slice(None), slice(None), 0), [100, 100, 100])

@with_ctx
def test_slice_reduce(ctx):
  x = expr.arange((TEST_SIZE, TEST_SIZE, TEST_SIZE), dtype=np.int)
  nx = np.arange(TEST_SIZE * TEST_SIZE * TEST_SIZE, dtype=np.int).reshape((TEST_SIZE, TEST_SIZE, TEST_SIZE))
  y = x[:, :, 0].sum()
  val = y.evaluate().glom()
  
  Assert.all_eq(val, nx[:, :, 0].sum())
  