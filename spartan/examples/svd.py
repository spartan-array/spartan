from spartan import expr, util, blob_ctx
import lanczos
import numpy as np
from spartan.array import distarray
from spartan.expr.shuffle import target_mapper

def svds(A, k=6):
  """Compute the largest k singular values/vectors for a sparse matrix.

  Parameters
  ----------
  A : sparse matrix
      Array to compute the SVD on, of shape (M, N)
  k : int, optional
      Number of singular values and vectors to compute.

 Returns
  -------
  u : ndarray, shape=(M, k)
      Unitary matrix having left singular vectors as columns.
  s : ndarray, shape=(k,)
      The singular values.
  vt : ndarray, shape=(k, N)
      Unitary matrix having right singular vectors as rows.  
  """
  AT = expr.transpose(A)
  d, u = lanczos.solve(AT, A, k)
  d, v =  lanczos.solve(A, AT, k)
  return u, np.sqrt(d), v.T
