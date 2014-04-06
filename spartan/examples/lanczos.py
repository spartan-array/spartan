import numpy as np
from spartan import expr, util
import math

def solve(A, AT, desiredRank):
  """Simple implementation of the ("http://en.wikipedia.org/wiki/Lanczos_algorithm")
  Lanczos algorithm for finding eigenvalues of a sparse matrix. We only parallelize
  the steps of multiplying the matrix with the vector since this step consumes most time 
  of the computation if your matrix is sparse enough. 
  
  This implementation uses EigenDecomposition to do the eigenvalue extraction from the 
  small (desiredRank x desiredRank) tridiagonal matrix.  Numerical stability is achieved 
  via brute-force: re-orthogonalization against all previous eigenvectors is computed 
  after every pass.
  
  The EigenDecomposition is completely sequential and written in Python. This is OK since
  the time complexity is O(desiredRank ^ 2) and the rank of sparse matrix should be way smaller
  than the size of matrix:
    
    desiredRank << M or N

  """
  desiredRank += 2

  n = A.shape[1]
  v_next = np.ones(n) / np.sqrt(n)
  v_prev = np.zeros(n)
  beta = np.zeros(desiredRank+1)
  beta[0] = 0
  alpha = np.zeros(desiredRank)

  # Since the disiredRank << size of matrix, so we keep
  # V in local memory for efficiency reason(It needs to be updated
  # for every iteration). 
  # If the case which V can't be fit in local memory occurs, 
  # you could turn it into spartan distributed array. 
  V = np.zeros((n, desiredRank))

  for i in range(0, desiredRank):
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
  
  # Create tridiag matrix with size (desiredRank X desiredRank)  
  tridiag = np.diag(alpha)
  for i in range(0, desiredRank-1):
    tridiag[i, i+1] = beta[i+1] 
    tridiag[i+1, i] = beta[i+1]
  
  # Get eigenvectors and eigenvalues of this tridiagonal matrix.  
  # The eigenvalues of this tridiagnoal matrix equals to the eigenvalues
  # of matrix dot(A, A.T.). We can get the eigenvectors of A by multiplying 
  # V with eigenvectors of this tridiagonal matrix.
  d, v = eig_decomposition(alpha, beta, tridiag)
  s = np.dot(V, v)
  return d[desiredRank-3::-1], s[:, desiredRank-3::-1] 


def eig_decomposition(alpha, beta, v):
  """Given a symmetric matrix, find its eigenvectors and eigenvalues
  using QL method(http://beige.ucs.indiana.edu/B673/node35.html).
  The time complexity is O(desiredRank ^ 2). desiredRank should be 
  small so this method will not be the bottleneck. Otherwise,
  we should rewrite this method in Cython.
  """
  n = alpha.shape[0]
  # d contains diagonal elements.
  d = alpha
  # e contains subdiagnal elements.
  e = beta[0:n]
  e[0:n-1] = e[1:n]
  e[n-1] = 0.0
  
  f = 0.0
  tst1 = 0.0
  # A very small number.
  eps = 0.000000000000001
  
  for l in xrange(0, n):
    fail_condition = False
    # Find small subdiagonal element
    tst1 = max(tst1, abs(d[l]) + abs(e[l]))
    m = l

    while m < n:
      if abs(e[m]) <= eps * tst1:
        break
      m += 1
    
    if m > l:
      while not fail_condition:
        g = d[l]
        p = (d[l+1] - g) / (2.0 * e[l])
        r = math.hypot(p, 1.0) 
        if p < 0.0:
          r = -r

        d[l] = e[l] / (p + r)
        d[l+1] = e[l] * (p + r)
        dl1 = d[l+1]
        h = g - d[l]
        for i in xrange(l+2, n):
          d[i] = d[i] - h
          
        f += h
        
        # Implicit QL transformation

        p = d[m]
        c = 1.0
        c2 = c
        c3 = c
        el1 = e[l+1]
        s = 0.0
        s2 = 0.0
        for i in range(m-1, l-1, -1):
          c3 = c2
          c2 = c
          s2 = s
          g = c * e[i]
          h = c * p
          r = math.hypot(p, e[i])
          e[i+1] = s * r
          s = e[i] / r
          c = p / r
          p = c * d[i] - s * g
          d[i+1] = h + s * (c * g + s * d[i])

          for k in xrange(0, n):
            h = v[k][i+1] 
            v[k][i+1] = s * v[k][i] + c * h
            v[k][i] = c * v[k][i] - s * h
          
        p = (-s) * s2 * c3 * el1 * e[l] / dl1
        e[l] = s * p
        d[l] = c * p
        
        if abs(e[l]) <= eps * tst1:
          fail_condition = True

    d[l] = d[l] + f
    e[l] = 0.0
  
  # Sort eigenvalues and corresponding vectors.
  arg_idx = np.argsort(d)[::-1]
  d = d[arg_idx]
  v = v[:,arg_idx]
  for i in range(n):
    v[:, i] /= np.linalg.norm(v[:, i])
  return d, v
