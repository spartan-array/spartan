from spartan.array import prims, compile_expr, distarray_backend, distarray
import numpy as np
import test_common
from spartan.util import Assert
 
def test_add2(master):
  a = distarray.DistArray.ones(master, (10, 10))
  b = distarray.DistArray.ones(master, (10, 10))
  a = prims.Value(a)
  b = prims.Value(b)
  
  map_prim = prims.Map([a, b], lambda a, b: a + b)
  c = distarray_backend.evaluate(master, map_prim)
  lc = c.glom()
  Assert.all_eq(lc, np.ones((10, 10)) + np.ones((10, 10)))

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

if __name__ == '__main__':
  test_common.run_cluster_tests(__file__)
