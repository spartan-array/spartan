import spartan
from spartan import expr, core
import numpy as np
from sys import stderr

def qr(Y):
  ''' Compute the thin qr factorization of a matrix.
  Factor the matrix Y as QR, where Q is orthonormal and R is
  upper-triangular.

  Parameters
  ----------
  Y: Spartan array of shape (M, K).
  
  Notes
  ----------
  Y'Y must fit in memory. Y is a Spartan array of shape (M, K).
  Since this QR decomposition is mainly used in Stochastic SVD,
  K will be the rank of the matrix of shape (M, N) and the assumption
  is that the rank K should be far less than M or N. 

  Returns
  -------
  Q : Spartan array of shape (M, K).
  R : Numpy array of shape (K, K).
  '''
  # Since the K should be far less than M. So the matrix multiplication
  # should be the bottleneck instead of local cholesky decomposition and 
  # finding inverse of R. So we just parallelize the matrix mulitplication.
  # If K is really large, we may consider using our Spartan cholesky 
  # decomposition, but for now, we use numpy version, it works fine.

  # YTY = Y'Y. YTY has shape of (K, K).
  YTY = expr.dot(expr.transpose(Y), Y).glom() 
  
  # Do cholesky decomposition and get R.
  R = np.linalg.cholesky(YTY).T

  # Find the inverse of R
  inv_R = np.linalg.inv(R)

  # Q = Y * inv(R)
  Q = expr.dot(Y, inv_R).force()

  return Q, R 
