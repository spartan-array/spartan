from spartan import expr, util
import test_common
from test_common import millis
import numpy as np
from datetime import datetime

#@test_common.with_ctx
#def test_pr(ctx):
def benchmark_sort(ctx, timer):
  A = expr.rand(10, 10, 10).force()
  T = expr.sort(A)
  print np.all(np.equal(T.glom(), np.sort(A.glom(), axis=None)))
  
if __name__ == '__main__':
  test_common.run(__file__)
