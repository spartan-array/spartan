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
    (av * bv)[:, -1].reshape((DIM, )) : Expr
        RHS vector extracted from input array
    """
    av = expr.arange(start = 2, stop = size + 2)
    bv = expr.arange(start = 4, stop = size + 4).reshape((size, 1))

    return av * bv, (av * bv)[:, -1:].reshape((size, ))

def jacobi_method(A, b, _iter = 100):
  """
  Iterative algorithm for approximating the solutions of a diagonally dominant system of linear equations. 

  Parameters
  ----------
  A : ndarray or Expr - 2d
      Input matrix
  b : ndarray or Expr - vector
      RHS vector
  _iter : int
      Times of iteration needed, default to be 100

 Returns
  -------
  result : Expr - vector
      Approximated solution.
  """
  #A = A = jacobi_init(DIM)
  #b = A[:, DIM-1:].reshape((DIM, ))

  util.Assert.eq(A.shape[0], b.shape[0])

  x = expr.zeros((A.shape[0],))

  D = expr.diag(A)
  R = A - expr.diagflat(D)

  for i in xrange(_iter):
    x = (b - expr.dot(R, x)) / D

  return x
