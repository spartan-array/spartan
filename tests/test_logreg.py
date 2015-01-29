from spartan.examples import logistic_regression
from spartan import expr, util
from spartan.config import FLAGS
import test_common
import time

N_EXAMPLES = 100
N_DIM = 3
ITERATION = 2

class TestLogisticRegression(test_common.ClusterTest):
  def test_logreg(self):
    FLAGS.opt_parakeet_gen = 0
    logistic_regression.run(N_EXAMPLES, N_DIM, ITERATION)

def benchmark_logreg(ctx, timer):
  print "#worker:", ctx.num_workers
  FLAGS.opt_parakeet_gen = 0
  N_EXAMPLES = 600000 * ctx.num_workers
  N_DIM = 512
  #N_EXAMPLES = 5000000 * 64
  x = expr.rand(N_EXAMPLES, N_DIM)
  y = expr.rand(N_EXAMPLES, 1)
  start = time.time()
  logistic_regression.logistic_regression(x, y, ITERATION)

  total = time.time() - start
  util.log_warn("time cost : %s s" % (total*1.0/ITERATION,))
  
if __name__ == '__main__':
  test_common.run(__file__)
