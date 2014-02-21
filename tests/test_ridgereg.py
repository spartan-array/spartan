from spartan.examples import ridge_regression
import test_common

N_EXAMPLES = 100
N_DIM = 3
ITERATION = 10

class TestRidgeRegression(test_common.ClusterTest):
  def test_ridgereg(self):
    ridge_regression.run(N_EXAMPLES, N_DIM, ITERATION)
