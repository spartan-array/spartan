import test_common
from spartan.examples.ssvd import qr 
from scipy import linalg 
import numpy as np
from numpy import absolute
from spartan import expr
from sys import stderr

class TestQR(test_common.ClusterTest):
  def test_qr(self):
    M = 100
    N = 10
    Y = expr.randn(M, N, tile_hint=(M/4, N)).force()
    
    Q1, R1 = qr(Y)
    Q1 = Q1.glom()
    Q2, R2 = linalg.qr(Y.glom(), mode="economic")
    
    assert np.allclose(np.absolute(Q1), np.absolute(Q2))
    assert np.allclose(np.absolute(R1), np.absolute(R2))
