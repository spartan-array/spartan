import math
import numpy as np
from scipy import linalg
from spartan import expr, util
from spartan.array import extent
from spartan.config import FLAGS


def _cholesky_dpotrf_mapper(extents, tiles):
  input = tiles[0]
  ex = extents[0]
  L, info = linalg.lapack.dpotrf(input, lower=1)
  return ex, L


def _cholesky_dtrsm_mapper(extents, tiles):
  input = tiles[0]
  A_kk = tiles[1]
  L, info = linalg.lapack.dtrtrs(A_kk, input.T, lower=1)
  return extents[0], L.T


def _cholesky_dsyrk_dgemm_mapper(extents, tiles):
  util.log_warn("dgemm %s" % str(extents))
  input = tiles[0]
  ex = extents[0]
  A_mk = tiles[1].T

  if ex.ul[0] == ex.ul[1] and ex.lr[0] == ex.lr[1]:
    # diag block
    return ex, linalg.blas.dsyrk(-1.0, A_mk, 1.0, input, lower=1)
  else:
    # other block
    A_lk = tiles[2]
    return ex, linalg.blas.dgemm(-1.0, A_lk, A_mk.T, 1.0, input)


def _zero_mapper(extents, tiles):
  return extents[0], np.zeros(tiles[0].shape, tiles[0].dtype)


def get_ex(i, j, step, array_shape):
  return extent.create((i*step, j*step), ((i+1)*step, (j+1)*step), array_shape)


def cholesky(A):
  '''
  Cholesky matrix decomposition.

  Args:
    A(Expr): matrix to be decomposed
  '''
  n = int(math.sqrt(FLAGS.num_workers))
  tile_size = A.shape[0] / n
  print n, tile_size
  for k in range(n):
    # A[k,k] = DPOTRF(A[k,k])
    diag_ex = get_ex(k, k, tile_size, A.shape)
    A = expr.map2(A, ((0, 1), ), fn=_cholesky_dpotrf_mapper,
                  shape=A.shape, update_region=diag_ex)

    if k == n - 1: break

    # A[l,k] = DTRSM(A[k,k], A[l,k]) l -> [k+1,n)
    col_ex = extent.create(((k+1)*tile_size, k*tile_size), (n*tile_size, (k+1)*tile_size), A.shape)
    A = expr.map2((A, A[diag_ex.to_slice()]), ((0, 1), None), fn=_cholesky_dtrsm_mapper,
                  shape=A.shape, update_region=col_ex)

    # A[m,m] = DSYRK(A[m,k], A[m,m]) m -> [k+1,n)
    # A[l,m] = DGEMM(A[l,k], A[m,k], A[l,m]) m -> [k+1,n) l -> [m+1,n)
    col_exs = list([extent.create((m*tile_size, m*tile_size), (n*tile_size, (m+1)*tile_size), A.shape) for m in range(k+1, n)])
    dgemm = A[:, (k * tile_size):((k + 1) * tile_size)]
    A = expr.map2((A, expr.transpose(dgemm), dgemm), ((0, 1), 1, 0),
                  fn=_cholesky_dsyrk_dgemm_mapper,
                  shape=A.shape, update_region=col_exs).optimized()

  # update the right corner to 0
  col_exs = list([extent.create((0, m*tile_size), (m*tile_size, (m+1)*tile_size), A.shape) for m in range(1, n)])
  A = expr.map2(A, ((0, 1), ), fn=_zero_mapper,
                shape=A.shape, update_region=col_exs)
  return A
