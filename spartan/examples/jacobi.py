import numpy as np
from spartan import expr, blob_ctx, util
import time

def jacobi_init(size):
    """
    Input array constructor

    Parameters
    ----------
    size : int
        Size of one dimension to array

    Returns
    ----------
    av * bv : spartan nd array
        Formatted input array to computation
    """
    av = expr.arange(start = 2, stop = size + 2)
    bv = expr.arange(start = 4, stop = size + 4).reshape((size, 1))

    return av * bv

def jacobi_method(base):
  """
  Iterative algorithm for approximating the solutions of a diagonally dominant system of linear equations. 

  Parameters
  ----------
  base : int
      Factor for the shape of array.
      e.g., if base is 100, then shape of array will be ((100 * num_workers), (100 * num_workers))

 Returns
  -------
  result : vector
      Approximated solution.
  """

    DIM = base * (blob_ctx.get().num_workers)

    A = A = self.jacobi_init(DIM)
    b = A[:, DIM-1:].reshape((DIM, ))
    #b = expr.randn(DIM)

    x = expr.zeros((DIM,))

    D = expr.diag(A)
    R = A - expr.diagflat(D)

    start = time.time()

    for i in xrange(100):
      x = (b - expr.dot(R, x)) / D

    result = x.glom()

    cost = time.time() - start

    util.log_info('cost =', cost)
    return result
