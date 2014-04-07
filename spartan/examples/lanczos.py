import numpy as np
from spartan import expr, util
import math

def solve(A, AT, desired_rank):
  '''
  A simple implementation of the Lanczos algorithm
  (http://en.wikipedia.org/wiki/Lanczos_algorithm) for eigenvalue computation.

  Like the Mahout implementation, only the matrix*vector step is parallelized.
  
  First we use lanczos method to turn the matrix into tridiagonoal form. Then
  we use numpy.linalg.eig function to extract the eigenvalues and eigenvectors 
  from the tridiagnonal matrix(desired_rank*desired_rank). Since desired_rank 
  should be smaller than the size of matrix, so we could it in local machine 
  efficiently. 
  '''
  
  # Calculate two more eigenvalues, but we only keep the largest desired_rank
  # one. Doing this to keep the result consistent with scipy.sparse.linalg.svds.
  desired_rank += 2

  n = A.shape[1]
  v_next = np.ones(n) / np.sqrt(n)
  v_prev = np.zeros(n)
  beta = np.zeros(desired_rank+1)
  beta[0] = 0
  alpha = np.zeros(desired_rank)

  # Since the disiredRank << size of matrix, so we keep
  # V in local memory for efficiency reason(It needs to be updated
  # for every iteration). 
  # If the case which V can't be fit in local memory occurs, 
  # you could turn it into spartan distributed array. 
  V = np.zeros((n, desired_rank))

  for i in range(0, desired_rank):
    util.log_info("Iter : %s", i)
    
    w = expr.dot(A, v_next.reshape(n, 1))
    w = expr.dot(AT, w).glom().reshape(n)

    alpha[i] = np.dot(w, v_next)
    w = w - alpha[i] * v_next - beta[i] * v_prev
    
    # Orthogonalize:
    for t in range(i):
      tmpa = np.dot(w, V[:, t])
      if tmpa == 0.0:
        continue
      w -= tmpa * V[:, t] 

    beta[i+1] = np.linalg.norm(w, 2) 
    v_prev = v_next
    v_next = w / beta[i+1]
    V[:, i] = v_prev
  
  # Create tridiag matrix with size (desired_rank X desired_rank)  
  tridiag = np.diag(alpha)
  for i in range(0, desired_rank-1):
    tridiag[i, i+1] = beta[i+1] 
    tridiag[i+1, i] = beta[i+1]
  
  # Get eigenvectors and eigenvalues of this tridiagonal matrix.  
  # The eigenvalues of this tridiagnoal matrix equals to the eigenvalues
  # of matrix dot(A, A.T.). We can get the eigenvectors of dot(A, A.T) 
  # by multiplying V with eigenvectors of this tridiagonal matrix.
  d, v = np.linalg.eig(tridiag) 
  
  # Sort eigenvalues and their corresponding eigenvectors 
  sorted_idx = np.argsort(d)[::-1]
  d = d[sorted_idx]
  v = v[:, sorted_idx]
  
  # Get the eigenvetors of dot(A, A.T)
  s = np.dot(V, v)
  return d[desired_rank-3::-1], s[:, desired_rank-3::-1] 
