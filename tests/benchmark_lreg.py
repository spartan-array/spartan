from spartan import util
from spartan import expr
from spartan.util import Assert
import numpy as np
import test_common


def benchmark_linear_regression(ctx, timer):
  N_EXAMPLES = 10 * 1000 * 1000 * ctx.num_workers
  N_DIM = 10
  x = expr.rand(N_EXAMPLES, N_DIM,
                tile_hint=(N_EXAMPLES / ctx.num_workers, N_DIM)).astype(np.float32)

  y = expr.rand(N_EXAMPLES, 1,
                tile_hint=(N_EXAMPLES / ctx.num_workers, 1)).astype(np.float32)

  w = np.random.rand(N_DIM, 1).astype(np.float32)

  x = expr.eager(x)
  y = expr.eager(y)

  def _step():
    yp = expr.dot(x, w)
    Assert.all_eq(yp.shape, y.shape)

    diff = x * (yp - y)
    grad = expr.sum(diff, axis=0).glom().reshape((N_DIM, 1))
    wprime = w - grad * 1e-6
    wprime.evaluate()

  for i in range(25):
    timer.time_op('linear-regression', _step)

if __name__ == '__main__':
  test_common.run(__file__)
