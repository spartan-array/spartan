from spartan import util
from spartan.array import prims, compile_expr, backend, distarray, expr
from spartan.util import Assert
import numpy as np
import test_common

distarray.TILE_SIZE = 10
 
def test_add2(master):
  a = distarray.ones(master, (10, 10))
  b = distarray.ones(master, (10, 10))
  a = prims.Value(a)
  b = prims.Value(b)
  
  map_prim = prims.MapTiles([a, b], lambda a, b: a + b)
  c = backend.evaluate(master, map_prim)
  lc = c.glom()
  Assert.all_eq(lc, np.ones((10, 10)) * 2)

def test_add3(master):
  a = distarray.ones(master, (10, 10))
  b = distarray.ones(master, (10, 10))
  c = distarray.ones(master, (10, 10))
  
  a = prims.Value(a)
  b = prims.Value(b)
  c = prims.Value(c)
  
  map_prim = prims.MapTiles([a, b, c], lambda a, b, c: a + b + c)
  d = backend.evaluate(master, map_prim)
  ld = d.glom()
  Assert.all_eq(ld, np.ones((10, 10)) * 3)
  
  
def test_compile_add2(master):
  a = expr.ones((10, 10))
  b = expr.ones((10, 10))
  Assert.all_eq((a + b).evaluate().glom(), np.ones((10, 10)) * 2)


def test_compile_add3(master):
  a = expr.ones((10, 10))
  b = expr.ones((10, 10))
  c = expr.ones((10, 10))
  Assert.all_eq((a + b + c).evaluate().glom(), np.ones((10, 10)) * 3)


def test_sum(master):
  a = distarray.ones(master, (10, 10))
  a = prims.Value(a)
  b = prims.Reduce(a, 0, 
                   lambda _: np.float, 
                   lambda ex, tile: np.sum(tile[:], axis=0), 
                   lambda a, b: a + b)
  c = backend.evaluate(master, b)
  lc = c.glom()
  Assert.all_eq(lc, np.ones((10,)) * 10)
  
def test_compile_sum(master):
  def _(axis):
    util.log('Testing sum over axis %s', axis)
    a = expr.ones((10, 10))
    b = a.sum(axis=axis)
    val = b.evaluate()
    Assert.all_eq(val.glom(), np.ones((10,10)).sum(axis))

  _(axis=0)
  _(axis=1)
  _(axis=None)
 
 
def test_compile_index(master):
  a = expr.arange((10, 10))
  b = expr.ones((10,))
  z = a[b]  
  val = z.evaluate()
  
  nx = np.arange(100).reshape(10, 10)
  ny = np.ones((10,), dtype=np.int)
  
  Assert.all_eq(val.glom(), nx[ny])
  
def test_slice(master):
  x = expr.arange((10, 10))
  z = x[5:8, 5:8]
  zc = compile_expr.compile_op(z)
  val = backend.evaluate(master, zc)
  nx = np.arange(100).reshape(10, 10)
  
  Assert.all_eq(val.glom(), nx[5:8, 5:8])
  
if __name__ == '__main__':
  test_common.run_cluster_tests(__file__)
