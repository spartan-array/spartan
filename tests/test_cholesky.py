from spartan import expr, util
from spartan.examples.cholesky import cholesky
import test_common
import numpy as np
from scipy import linalg
from datetime import datetime

def millis(t1, t2):
  dt = t2 - t1
  ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
  return ms

#@test_common.with_ctx
#def test_pr(ctx):
def benchmark_cholesky(ctx, timer):
  print "#worker:", ctx.num_workers
  ARRAY_SIZE = 100 * ctx.num_workers
  A = expr.randn(ARRAY_SIZE, ARRAY_SIZE, tile_hint=(ARRAY_SIZE/ctx.num_workers, ARRAY_SIZE/ctx.num_workers))
  A = expr.dot(A, expr.transpose(A))
  
  L = cholesky(A, ctx.num_workers).glom()
  assert np.all(np.isclose(A.glom(), np.dot(L, L.T.conj())))
  
if __name__ == '__main__':
  test_common.run(__file__)
