from spartan import util
from spartan.array import prims, compile_expr, backend, distarray, expr, extent
from spartan.util import Assert
import math
import numpy as np
import test_common

TEST_SIZE = 10000
def benchmark_slice_reduce(ctx):
  x = expr.arange((TEST_SIZE, TEST_SIZE))
  y = x[200:300]
  z = y.sum()
  zc = compile_expr.compile(z)
  val = backend.evaluate(ctx, zc)
  print val.glom()
  
  
if __name__ == '__main__':
  test_common.run_cluster_tests(__file__)