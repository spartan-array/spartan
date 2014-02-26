from spartan.examples import logistic_regression
from spartan import expr
import test_common

N_EXAMPLES = 100
N_DIM = 3
ITERATION = 10

class TestLogisticRegression(test_common.ClusterTest):
  def test_logreg(self):
    logistic_regression.run(N_EXAMPLES, N_DIM, ITERATION)

def benchmark_logreg(ctx, timer):
  print "#worker:", ctx.num_workers
  N_EXAMPLES = 1000000 * ctx.num_workers
  ITERATION = 100
  x = expr.eager(expr.rand(N_EXAMPLES, N_DIM, tile_hint=(N_EXAMPLES / ctx.num_workers, N_DIM)))
  y = expr.eager(expr.rand(N_EXAMPLES, 1, tile_hint=(N_EXAMPLES / ctx.num_workers, 1)))
  logistic_regression.logistic_regression(x, y, ITERATION)
  
if __name__ == '__main__':
  test_common.run(__file__)