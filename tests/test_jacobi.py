from spartan.examples import jacobi
from spartan import expr, util, blob_ctx
import test_common
import time

base = 2000
ITERATION = 10


class TestJacobiMethod(test_common.ClusterTest):
  def test_jacobi(self):
    global base
    A, b = jacobi.jacobi_init(base * blob_ctx.get().num_workers)
    jacobi.jacobi_method(A, b, 10).glom()


def benchmark_jacobi(ctx, timer):
  global base, ITERATION
  util.log_warn('util.log_warn: %s', ctx.num_workers)

  A, b = jacobi.jacobi_init(base * ctx.num_workers)
  A, b = A.evaluate(), b.evaluate()

  start = time.time()

  result = jacobi.jacobi_method(A, b, ITERATION).glom()

  cost = time.time() - start

  util.log_info('\nresult =\n%s', result)
  util.log_warn('time cost: %s s', cost)
  util.log_warn('cost per iteration: %s s\n', cost / ITERATION)

if __name__ == '__main__':
  test_common.run(__file__)
