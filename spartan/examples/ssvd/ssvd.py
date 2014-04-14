import spartan
from spartan import core, expr, util, blob_ctx
import numpy as np

def solve(A, k):
  """
  Stochastic SVD.

  Parameters
  ----------
  A : spartan matrix
      Array to compute the SVD on, of shape (M, N)
  k : int, optional
      Number of singular values and vectors to compute.

  The operations include matrix multiplication and QR decomposition.
  We parallelize both of them.
  """
  r = A.tile_shape()[0]
  ctx = blob_ctx.get()
  Omega = expr.randn(A.shape[1], k)
  r = A.shape[0] / ctx.num_workers
  Y = expr.dot(A, Omega, tile_hint=(r, k)).force()
  
  Q, R = qr(Y)
  
  B = expr.dot(expr.transpose(Q), A).glom()
  BTB = np.dot(B, B.T)
  D, U_ = eig(BTB)
  D = np.sqrt(D)
  U = np.dot(Q.glom(), U_)

  V = np.dot(np.dot(B.T, U_), np.diag(np.ones(D.shape[0]) / D))
  return U_, D, V 
