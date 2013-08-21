from spartan.array import prims, compile_expr, distarray_backend, distarray,\
  expr
import numpy as np
import test_common
from spartan.util import Assert

distarray.TILE_SIZE = 10
 
def test_add2(master):
  a = distarray.DistArray.ones(master, (10, 10))
  b = distarray.DistArray.ones(master, (10, 10))
  a = prims.Value(a)
  b = prims.Value(b)
  
  map_prim = prims.Map([a, b], lambda a, b: a + b)
  c = distarray_backend.evaluate(master, map_prim)
  lc = c.glom()
  Assert.all_eq(lc, np.ones((10, 10)) * 2)

def test_add3(master):
  a = distarray.DistArray.ones(master, (10, 10))
  b = distarray.DistArray.ones(master, (10, 10))
  c = distarray.DistArray.ones(master, (10, 10))
  
  a = prims.Value(a)
  b = prims.Value(b)
  c = prims.Value(c)
  
  map_prim = prims.Map([a, b, c], lambda a, b, c: a + b + c)
  d = distarray_backend.evaluate(master, map_prim)
  ld = d.glom()
  Assert.all_eq(ld, np.ones((10, 10)) * 3)
  
  
def test_compile_add2(master):
  a = distarray.DistArray.ones(master, (10, 10))
  b = distarray.DistArray.ones(master, (10, 10))
  
  a = expr.lazify(a)
  b = expr.lazify(b)
  c = a + b
  compiled_expr = compile_expr.compile_op(c)
  cval = distarray_backend.evaluate(master, compiled_expr)
  Assert.all_eq(cval.glom(), np.ones((10, 10)) * 2)


def test_compile_add3(master):
  a = distarray.DistArray.ones(master, (10, 10))
  b = distarray.DistArray.ones(master, (10, 10))
  c = distarray.DistArray.ones(master, (10, 10))
  
  a = expr.lazify(a)
  b = expr.lazify(b)
  c = expr.lazify(c)
  d = a + b + c
  compiled_expr = compile_expr.compile_op(d)
  dval = distarray_backend.evaluate(master, compiled_expr)
  Assert.all_eq(dval.glom(), np.ones((10, 10)) * 3)

if __name__ == '__main__':
  test_common.run_cluster_tests(__file__)
