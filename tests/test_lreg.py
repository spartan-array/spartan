from spartan.examples import linear_regression
import test_common

N_EXAMPLES = 100
N_DIM = 3
ITERATION = 100

class TestLinearRegression(test_common.ClusterTest):
  def test_lreg(self):
    linear_regression.run(N_EXAMPLES, N_DIM, ITERATION)
