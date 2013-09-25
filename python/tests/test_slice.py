from spartan import util
from spartan.array import prims, compile_expr, backend, distarray, expr, extent
from spartan.util import Assert
import math
import numpy as np
import test_common

DIM = 100
distarray.TILE_SIZE = DIM ** 2 / 4

def test_slice_get(ctx):
  x = expr.arange((DIM, DIM))
  z = x[5:8, 5:8]
  zc = compile_expr.compile(z)
  val = backend.evaluate(ctx, zc)
  nx = np.arange(DIM*DIM).reshape(DIM, DIM)
  Assert.all_eq(val.glom(), nx[5:8, 5:8])

def add_one_tile(tiles):
  return tiles[0] + 1

def test_slice_map_tiles(ctx):
  x = expr.arange((DIM, DIM))
  z = x[5:8, 5:8]
  z = expr.map_tiles(z, add_one_tile) 
  zc = compile_expr.compile(z)
  val = backend.evaluate(ctx, zc)
  nx = np.arange(DIM*DIM).reshape(DIM, DIM)
  
  Assert.all_eq(val.glom(), nx[5:8, 5:8] + 1)

def add_one_extent(inputs, ex):
  util.log('Mapping: %s', ex)
  return (ex, inputs[0][ex] + 1)

def test_slice_map_extents(ctx):
  x = expr.arange((DIM, DIM))
  z = x[5:8, 5:8]
  z = expr.map_extents(z, add_one_extent) 
  zc = compile_expr.compile(z)
  val = backend.evaluate(ctx, zc)
  nx = np.arange(DIM*DIM).reshape(DIM, DIM)
  
  Assert.all_eq(val.glom(), nx[5:8, 5:8] + 1)
  
  
def test_slice_map_tiles2(ctx):
  x = expr.arange((10, 10, 10), dtype=np.int)
  nx = np.arange(10 * 10 * 10, dtype=np.int).reshape((10, 10, 10))
  
  y = x[:, :, 0]
  z = expr.map_tiles(y, lambda tiles: tiles[0] + 13)
  val = z.evaluate().glom()
  
  Assert.all_eq(val, nx[:, :, 0] + 13)
  
def test_from_slice(ctx):
  print extent.from_slice((slice(None), slice(None), 0), [100, 100, 100])
  
if __name__ == '__main__':
  test_common.run_cluster_tests(__file__)