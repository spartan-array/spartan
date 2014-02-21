from spartan.examples import logistic_regression
import test_common

N_EXAMPLES = 100
N_DIM = 3
ITERATION = 10

class TestLogisticRegression(test_common.ClusterTest):
  def test_logreg(self):
    logistic_regression.run(N_EXAMPLES, N_DIM, ITERATION)
