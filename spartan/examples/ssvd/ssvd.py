import spartan
from spartan import core, expr, util, blob_ctx
import numpy as np
from .qr import qr

def svd(A, k=None):
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

  Returns
  --------
  U : Spartan array of shape (M, k)
  S : numpy array of shape (k,)
  V : numpy array of shape (k, k)
  """
  if k is None: k = A.shape[1]

  Omega = expr.randn(A.shape[1], k)

  Y = expr.dot(A, Omega)
  
  Q, R = qr(Y)
  
  B = expr.dot(expr.transpose(Q), A)
  BTB = expr.dot(B, expr.transpose(B)).optimized().glom()

  S, U_ = np.linalg.eig(BTB)
  S = np.sqrt(S)

  # Sort by eigen values from large to small
  si = np.argsort(S)[::-1]
  S = S[si]
  U_ = U_[:, si]

  U = expr.dot(Q, U_).optimized().force()
  V = np.dot(np.dot(expr.transpose(B).optimized().glom(), U_), np.diag(np.ones(S.shape[0]) / S))
  return U, S, V.T 
