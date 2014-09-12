import test_common
import numpy as np
from spartan import expr, util
import time

DIM = 40000

class Jacobi_Test(test_common.ClusterTest):
  def jacobi_init(self, size):
    av = expr.arange(start = 2, stop = size+2)
    bv = expr.arange(start = 4, stop = size+4).reshape((size,1))

    return av * bv

  def test_jacobi(self):
    global DIM
    A = self.jacobi_init(DIM)
    b = A[:,DIM-1:].reshape((DIM, ))
    x = expr.zeros((DIM,))

    D = expr.diag(A)
    R = A - expr.diag(D)

    start = time.time()

    for i in xrange(100):
      x = (b - expr.dot(R, x))/D

    result = x.glom()

    cost = time.time() - start
    util.log_info('cost = %s', cost)
