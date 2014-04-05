import math
import numpy as np
from scipy import linalg
from spartan import expr, util
from spartan.array import extent

def _cholesky_dpotrf_mapper(input, array, ex):
  L,info = linalg.lapack.dpotrf(input, lower=1)
  return L

def _cholesky_dtrsm_mapper(input, array, ex, diag_ex):
  A_kk = array.fetch(diag_ex)
  #L,info = linalg.lapack.dtrtrs(A_kk, input, lower=1)
  return np.dot(input, linalg.inv(A_kk.T))

def _cholesky_dsyrk_dgemm_mapper(input, array, ex, k):
  
  mk_ex = extent.create((ex.ul[1], k*input.shape[1]), (ex.lr[1], (k+1)*input.shape[1]), array.shape)
  A_mk = array.fetch(mk_ex)
  
  if ex.ul[0] == ex.ul[1] and ex.lr[0] == ex.lr[1]:
    # diag block
    return linalg.blas.dsyrk(-1.0, A_mk, 1.0, input, lower=1)
  else:
    # other block
    lk_ex = extent.create((ex.ul[0], k*input.shape[1]), (ex.lr[0], (k+1)*input.shape[1]), array.shape)
    A_lk = array.fetch(lk_ex)
    return linalg.blas.dgemm(-1.0, A_lk, A_mk.T, 1.0, input)

def get_ex(i, j, step, array_shape):
  return extent.create((i*step, j*step), ((i+1)*step, (j+1)*step), array_shape)

def cholesky(A):
  '''
  Cholesky matrix decomposition.
 
  Args:
    A(Expr): matrix to be decomposed
  '''
 
  A = expr.force(A)
  n = int(math.sqrt(len(A.tiles)))
  tile_size = A.shape[0] / n
  for k in range(n):
    # A[k,k] = DPOTRF(A[k,k])
    diag_ex = get_ex(k, k, tile_size, A.shape)
    A = expr.region_map(A, diag_ex, _cholesky_dpotrf_mapper)
    
    if k == n - 1: break
    
    # A[l,k] = DTRSM(A[k,k], A[l,k]) l -> [k+1,n)
    col_ex = extent.create(((k+1)*tile_size, k*tile_size),(n*tile_size, (k+1)*tile_size), A.shape)
    A = expr.region_map(A, col_ex, _cholesky_dtrsm_mapper, fn_kw=dict(diag_ex=diag_ex))
    
    # A[m,m] = DSYRK(A[m,k], A[m,m]) m -> [k+1,n)
    # A[l,m] = DGEMM(A[l,k], A[m,k], A[l,m]) m -> [k+1,n) l -> [m+1,n)
    col_exs = list([extent.create((m*tile_size, m*tile_size), (n*tile_size, (m+1)*tile_size), A.shape) for m in range(k+1,n)])
    A = expr.region_map(A, col_exs, _cholesky_dsyrk_dgemm_mapper, fn_kw=dict(k=k))
  
  
  # update the right corner to 0
  col_exs = list([extent.create((0, m*tile_size),(m*tile_size, (m+1)*tile_size),A.shape) for m in range(1,n)])
  A = expr.region_map(A, col_exs, lambda input, array, ex: np.zeros(input.shape, input.dtype))
  return A
        
