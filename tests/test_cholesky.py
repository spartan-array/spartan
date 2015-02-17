from spartan import expr, util
from spartan.examples.cholesky import cholesky
import test_common
from test_common import millis
import numpy as np
from scipy import linalg
import math
from spartan.expr.operator.write_array import from_numpy
from datetime import datetime


#@test_common.with_ctx
#def test_pr(ctx):
def benchmark_cholesky(ctx, timer):
  print "#worker:", ctx.num_workers

  #n = int(math.pow(ctx.num_workers, 1.0 / 3.0))
  n = int(math.sqrt(ctx.num_workers))
  #ARRAY_SIZE = 1600 * 4
  ARRAY_SIZE = 900 * n

  util.log_warn('prepare data!')
  #A = np.random.randn(ARRAY_SIZE, ARRAY_SIZE)
  #A = np.dot(A, A.T)

  A = expr.randn(ARRAY_SIZE, ARRAY_SIZE)
  A = expr.dot(A, expr.transpose(A))

  util.log_warn('begin cholesky!')
  t1 = datetime.now()
  L = cholesky(A).optimized().glom()
  t2 = datetime.now()
  #assert np.all(np.isclose(A.glom(), np.dot(L, L.T.conj())))
  cost_time = millis(t1, t2)
  print "total cost time:%s ms, per iter cost time:%s ms" % (cost_time, cost_time/n)

if __name__ == '__main__':
  test_common.run(__file__)
