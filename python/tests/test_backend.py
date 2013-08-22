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
  
  map_prim = prims.Map([a, b], lambda a, b: a + b)
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
  
  map_prim = prims.Map([a, b, c], lambda a, b, c: a + b + c)
  d = backend.evaluate(master, map_prim)
  ld = d.glom()
  Assert.all_eq(ld, np.ones((10, 10)) * 3)
  
  
def test_compile_add2(master):
  a = distarray.ones(master, (10, 10))
  b = distarray.ones(master, (10, 10))
  
  a = expr.lazify(a)
  b = expr.lazify(b)
  c = a + b
  compiled_expr = compile_expr.compile_op(c)
  cval = backend.evaluate(master, compiled_expr)
  Assert.all_eq(cval.glom(), np.ones((10, 10)) * 2)


def test_compile_add3(master):
  a = distarray.ones(master, (10, 10))
  b = distarray.ones(master, (10, 10))
  c = distarray.ones(master, (10, 10))
  
  a = expr.lazify(a)
  b = expr.lazify(b)
  c = expr.lazify(c)
  d = a + b + c
  compiled_expr = compile_expr.compile_op(d)
  dval = backend.evaluate(master, compiled_expr)
  Assert.all_eq(dval.glom(), np.ones((10, 10)) * 3)

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
    a = distarray.ones(master, (10, 10))
    a = expr.lazify(a)
    b = a.sum(axis=axis)
    c = compile_expr.compile_op(b)
    val = backend.evaluate(master, c)
    Assert.all_eq(val.glom(), np.ones((10,10)).sum(axis))

  _(axis=0)
  _(axis=1)
  _(axis=None)
  
  
if __name__ == '__main__':
  test_common.run_cluster_tests(__file__)
