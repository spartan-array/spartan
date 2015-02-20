import test_common
from spartan.examples.ssvd import qr
from scipy import linalg
import numpy as np
from numpy import absolute
from spartan import expr
from sys import stderr
from test_common import millis
from datetime import datetime


class TestQR(test_common.ClusterTest):
  def test_qr(self):
    M = 100
    N = 10
    Y = expr.randn(M, N, tile_hint=(M / 4, N)).evaluate()

    Q1, R1 = qr(Y)
    Q1 = Q1.glom()
    Q2, R2 = linalg.qr(Y.glom(), mode="economic")

    assert np.allclose(np.absolute(Q1), np.absolute(Q2))
    assert np.allclose(np.absolute(R1), np.absolute(R2))


def benchmark_qr(ctx, timer):
  M = 1280
  N = 1280
  Y = np.random.randn(M, N)
  Y = expr.from_numpy(Y)
  #Y = expr.randn(M, N)

  t1 = datetime.now()
  Q, R = qr(Y)
  t2 = datetime.now()
  cost_time = millis(t1, t2)

  print "total cost time:%s ms" % (cost_time)

if __name__ == '__main__':
  test_common.run(__file__)
