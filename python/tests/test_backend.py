from spartan import util
from spartan.array import prims, compile_expr, backend, distarray, expr, extent
from spartan.util import Assert
import math
import numpy as np
import test_common

TEST_SIZE = 1000
distarray.TILE_SIZE = TEST_SIZE ** 2 / 4
 
def test_add2(ctx):
  a = distarray.ones(ctx, (TEST_SIZE, TEST_SIZE))
  b = distarray.ones(ctx, (TEST_SIZE, TEST_SIZE))
  a = prims.Value(a)
  b = prims.Value(b)
  
  map_prim = prims.MapTiles([a, b], lambda v: v[0] + v[1])
  c = backend.evaluate(ctx, map_prim)
  lc = c.glom()
  Assert.all_eq(lc, np.ones((TEST_SIZE, TEST_SIZE)) * 2)

def test_add3(ctx):
  a = distarray.ones(ctx, (TEST_SIZE, TEST_SIZE))
  b = distarray.ones(ctx, (TEST_SIZE, TEST_SIZE))
  c = distarray.ones(ctx, (TEST_SIZE, TEST_SIZE))
  
  a = prims.Value(a)
  b = prims.Value(b)
  c = prims.Value(c)
  
  map_prim = prims.MapTiles([a, b, c], lambda v: v[0] + v[1] + v[2])
  d = backend.evaluate(ctx, map_prim)
  ld = d.glom()
  Assert.all_eq(ld, np.ones((TEST_SIZE, TEST_SIZE)) * 3)
  
  
def test_compile_add2(ctx):
  a = expr.ones((TEST_SIZE, TEST_SIZE))
  b = expr.ones((TEST_SIZE, TEST_SIZE))
  Assert.all_eq((a + b).evaluate().glom(), np.ones((TEST_SIZE, TEST_SIZE)) * 2)


def test_compile_add3(ctx):
  a = expr.ones((TEST_SIZE, TEST_SIZE))
  b = expr.ones((TEST_SIZE, TEST_SIZE))
  c = expr.ones((TEST_SIZE, TEST_SIZE))
  Assert.all_eq((a + b + c).evaluate().glom(), np.ones((TEST_SIZE, TEST_SIZE)) * 3)

def test_compile_add_many(ctx):
  a = expr.ones((TEST_SIZE, TEST_SIZE))
  b = expr.ones((TEST_SIZE, TEST_SIZE))
  Assert.all_eq((a + b + a + b + a + b + a + b + a + b).evaluate().glom(), np.ones((TEST_SIZE, TEST_SIZE)) * 10)
  

def test_sum(ctx):
  a = distarray.ones(ctx, (TEST_SIZE, TEST_SIZE))
  a = prims.Value(a)
  b = prims.Reduce(a, 0, 
                   lambda _: np.float, 
                   lambda ex, tile: np.sum(tile[:], axis=0), 
                   lambda a, b: a + b)
  c = backend.evaluate(ctx, b)
  lc = c.glom()
  Assert.all_eq(lc, np.ones((TEST_SIZE,)) * TEST_SIZE)
  
def test_compile_sum(ctx):
  def _(axis):
    util.log('Testing sum over axis %s', axis)
    a = expr.ones((TEST_SIZE, TEST_SIZE))
    b = a.sum(axis=axis)
    val = b.evaluate()
    Assert.all_eq(val.glom(), np.ones((TEST_SIZE,TEST_SIZE)).sum(axis))

  _(axis=0)
  _(axis=1)
  _(axis=None)
 
 
def test_compile_index(ctx):
  a = expr.arange((TEST_SIZE, TEST_SIZE))
  b = expr.ones((10,))
  z = a[b]  
  val = z.evaluate()
  
  nx = np.arange(TEST_SIZE * TEST_SIZE).reshape(TEST_SIZE, TEST_SIZE)
  ny = np.ones((10,), dtype=np.int)
  
  Assert.all_eq(val.glom(), nx[ny])
  
if __name__ == '__main__':
  test_common.run_cluster_tests(__file__)