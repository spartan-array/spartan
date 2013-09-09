from spartan import util
from spartan.array import prims, compile_expr, backend, distarray, expr, extent
from spartan.util import Assert
import math
import numpy as np
import test_common

DIM = 1000
distarray.TILE_SIZE = DIM ** 2 / 4
 
def test_add2(ctx):
  a = distarray.ones(ctx, (DIM, DIM))
  b = distarray.ones(ctx, (DIM, DIM))
  a = prims.Value(a)
  b = prims.Value(b)
  
  map_prim = prims.MapTiles([a, b], lambda v: v[0] + v[1])
  c = backend.evaluate(ctx, map_prim)
  lc = c.glom()
  Assert.all_eq(lc, np.ones((DIM, DIM)) * 2)

def test_add3(ctx):
  a = distarray.ones(ctx, (DIM, DIM))
  b = distarray.ones(ctx, (DIM, DIM))
  c = distarray.ones(ctx, (DIM, DIM))
  
  a = prims.Value(a)
  b = prims.Value(b)
  c = prims.Value(c)
  
  map_prim = prims.MapTiles([a, b, c], lambda v: v[0] + v[1] + v[2])
  d = backend.evaluate(ctx, map_prim)
  ld = d.glom()
  Assert.all_eq(ld, np.ones((DIM, DIM)) * 3)
  
  
def test_compile_add2(ctx):
  a = expr.ones((DIM, DIM))
  b = expr.ones((DIM, DIM))
  Assert.all_eq((a + b).evaluate().glom(), np.ones((DIM, DIM)) * 2)


def test_compile_add3(ctx):
  a = expr.ones((DIM, DIM))
  b = expr.ones((DIM, DIM))
  c = expr.ones((DIM, DIM))
  Assert.all_eq((a + b + c).evaluate().glom(), np.ones((DIM, DIM)) * 3)

def test_compile_add_many(ctx):
  a = expr.ones((DIM, DIM))
  b = expr.ones((DIM, DIM))
  Assert.all_eq((a + b + a + b + a + b + a + b + a + b).evaluate().glom(), np.ones((DIM, DIM)) * 10)
  

def test_sum(ctx):
  a = distarray.ones(ctx, (DIM, DIM))
  a = prims.Value(a)
  b = prims.Reduce(a, 0, 
                   lambda _: np.float, 
                   lambda ex, tile: np.sum(tile[:], axis=0), 
                   lambda a, b: a + b)
  c = backend.evaluate(ctx, b)
  lc = c.glom()
  Assert.all_eq(lc, np.ones((DIM,)) * DIM)
  
def test_compile_sum(ctx):
  def _(axis):
    util.log('Testing sum over axis %s', axis)
    a = expr.ones((DIM, DIM))
    b = a.sum(axis=axis)
    val = b.evaluate()
    Assert.all_eq(val.glom(), np.ones((DIM,DIM)).sum(axis))

  _(axis=0)
  _(axis=1)
  _(axis=None)
 
 
def test_compile_index(ctx):
  a = expr.arange((DIM, DIM))
  b = expr.ones((10,))
  z = a[b]  
  val = z.evaluate()
  
  nx = np.arange(DIM * DIM).reshape(DIM, DIM)
  ny = np.ones((10,), dtype=np.int)
  
  Assert.all_eq(val.glom(), nx[ny])
  
def test_slice(ctx):
  x = expr.arange((DIM, DIM))
  z = x[5:8, 5:8]
  zc = compile_expr.compile(z)
  val = backend.evaluate(ctx, zc)
  nx = np.arange(DIM*DIM).reshape(DIM, DIM)
  
  Assert.all_eq(val.glom(), nx[5:8, 5:8])
  
def test_linear_regression(ctx):
  N_EXAMPLES = 2 * 1000 * 1000 * ctx.num_workers()
  N_DIM = 10
  distarray.TILE_SIZE = N_EXAMPLES / (4 * ctx.num_workers()) 
  x = expr.lazify(expr.rand(N_EXAMPLES, N_DIM).evaluate())
  y = expr.lazify(expr.rand(N_EXAMPLES, 1).evaluate())
  w = np.random.rand(N_DIM, 1)
  
  util.log('INIT DONE')
  for i in range(10):
    util.log('START')
    yp = expr.map_extents(x, expr._dot, w=w)
    diff = x * (yp - y)
    util.log('DIFF')
    diff.evaluate()
    util.log('DONE')
    grad = expr.sum(diff, axis=0).glom().reshape((N_DIM, 1))
    w = w + grad * 0.0001
    
    util.log('END')
  
if __name__ == '__main__':
  test_common.run_cluster_tests(__file__)
