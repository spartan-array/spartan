from spartan.examples import ridge_regression
import test_common
from test_common import millis
from spartan import expr, util
import time

N_EXAMPLES = 100
N_DIM = 3
ITERATION = 10

class TestRidgeRegression(test_common.ClusterTest):
  def test_ridgereg(self):
    ridge_regression.run(N_EXAMPLES, N_DIM, ITERATION)

def benchmark_ridgereg(ctx, timer):
  print "#worker:", ctx.num_workers
  #N_EXAMPLES = 100000000 * ctx.num_workers
  N_EXAMPLES = 90000000 * ctx.num_workers
  x = expr.rand(N_EXAMPLES, N_DIM)
  y = expr.rand(N_EXAMPLES, 1)
  start = time.time() 
  ridge_regression.ridge_regression(x, y, 1, ITERATION)
  
  total = time.time() - start
  util.log_warn("time cost : %s s" % (total*1.0/ITERATION,))
  
if __name__ == '__main__':
  test_common.run(__file__)
 
