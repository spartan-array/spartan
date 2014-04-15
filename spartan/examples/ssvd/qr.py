"""
QR Decomposition implemented on givens rotations:

Given a Y matrix, find the QR decomposition of this matrix.
We divide Y into blocks Y0, Y1, Y2...Yn and then compute QR
decomposition on each block of Y in parallel. After that, we 
get a list of Qs and Rs. We find if we merge these Qs and Rs 
sequentially, it takes a lot of time. Instead, we do a tree 
reduction to do it in parallel. The number of synronization 
in reduction is log(N), N is the number of blocks.

Tree reduction:
Q1  Q2  Q3  Q4 
|  /    |  /
| /     | /
Q1'     Q2' 
|      /
|   /
|/
Q...

"""
import spartan
from spartan import core, expr, util, blob_ctx
from spartan.array import extent
import numpy as np
from scipy.linalg import qr as skqr
import parakeet

def givens(a, b):
  a = float(a)
  b = float(b)
  if b == 0:
    return 1, 0
  if abs(b) > abs(a):
    tau = a / b
    u = np.sqrt(1 + tau * tau)
    if b < 0:
      u = -u
    s = -1 / u
    c = -s * tau
  else:
    tau = b / a
    u = np.sqrt(1 + tau * tau)
    if a < 0:
      u = -u
    c = 1 / u
    s = -c * tau
  return c, s 

def _qr_mapper(ex, Y, k, r, global_R, global_Q):
  '''Calculate Q,R decomposition on each block of Y.
  We divide Y into blocks Y1, Y2, .... YN. And then calculate
  the Q, R decompostion on each block of Y. Store the result
  to distributed array. It will be merged by another kernel.
  '''
  local_y = Y.fetch(ex)
  
  Q, R  = skqr(local_y, mode="economic")
  res = core.LocalKernelResult()
  res.result = None #(qq.T, rr)
  
  global_Q[:, ex.ul[0]:ex.lr[0]] = Q.T
  st_row = ex.ul[0] / r * k
  global_R[st_row : st_row + k, :] = R 
  return res


def _reduce_mapper(ex, Q, R, r, round):
  """ Reduce kernel reduces current Q, R with next pair of Q, R """
  #Index of this worker
  idx = ex.ul[0]
  k = Q.shape[0]
  qr =  2 ** (round - 1) * r  
  # Fetch first Q from global spartan array.
  local_Q1 = Q.fetch(extent.create(ul=(0, 2 * idx * qr), 
                                  lr=(Q.shape[0], 2 * idx * qr + qr), 
                                  array_shape=Q.shape))

  # Fetch second Q from global spartan array.
  local_Q2 = Q.fetch(extent.create(ul=(0, 2 * idx * qr + qr), 
                                  lr=(Q.shape[0], 2 * idx * qr + 2 * qr), 
                                  array_shape=Q.shape))
  
  # Fetch first R from global spartan array.
  local_R1 = R.fetch(extent.create(ul=(2 ** round * idx * k, 0),
                                  lr=(2 ** round * idx * k + k, R.shape[1]),
                                  array_shape=R.shape))

  # Fetch second R from global spartan array.
  local_R2 = R.fetch(extent.create(ul=((2 ** round * idx + 2 ** (round - 1)) * k, 0),
                                  lr=((2 ** round * idx + 2 ** (round - 1)) * k + k, R.shape[1]),
                                  array_shape=R.shape))
  
  # Merge two Q, R pairs.
  new_Q, new_R = computeQHatSeq(qs=[local_Q1, local_Q2],
                                rs=[local_R1, local_R2], k=k, r=qr) 
  
  # Store back to global Q, R
  Q[:, 2 * idx * qr : 2 * idx * qr + 2 * qr] = new_Q.T
  R[2 ** round * idx * k : 2 ** round * idx * k + k, :] = new_R
  result = core.LocalKernelResult()
  result.result = None
  return result


@parakeet.jit
def apply_givens_in_place(c, s, row1, row2, offset, len):
  n = offset + len
  for j in range(offset, n):
    tau1 = row1[j]
    tau2 = row2[j]
    row1[j] = c * tau1 - s * tau2
    row2[j] = s * tau1 + c * tau2

def mergeR(R1, R2, k):
  for v in range(0, k):
    for u in range(v, k):
      c, s = givens(R1[u, u], R2[u-v, u])
      apply_givens_in_place(c, s, R1[u], R2[u-v], u, k-u)
  return R1

def merge_R_on_Q(R1, R2, Q1, Q2, k):
  r = Q1.shape[1]
  assert R1.shape[0] == k
  assert Q1.shape[0] == k
  assert Q2.shape[0] == k 
  assert Q2.shape[1] == r
  
  for v in range(0, k):
    for u in range(v, k):
      c, s = givens(R1[u, u], R2[u-v, u])
      apply_givens_in_place(c, s, R1[u], R2[u-v], u, k-u)
      apply_givens_in_place(c, s, Q1[u], Q2[u-v], 0, r)

def merge_QR_up(Q, R1, R2):
  k = Q.shape[0]
  r = Q.shape[1]
  QTemp = np.zeros(Q.shape)
  merge_R_on_Q(R1, R2, Q, QTemp, k)
  return Q

def merge_QR_down(R1, Q, R2):
  k = Q.shape[0]
  r = Q.shape[1]
  QTemp = np.zeros(Q.shape)
  merge_R_on_Q(R1, R2, QTemp, Q, k)
  return QTemp

def computeQtHat(Q, i, rs):
  rs_ = [r.copy() for r in rs]
  rTilde = rs_[0]

  for j in range(1, i):
    mergeR(rTilde, rs_[j])

  if i > 0:
    Q = merge_QR_down(rTilde, Q, rs_[i])

  for t in range(i+1, len(rs)):
    Q = merge_QR_up(Q, rTilde, rs_[t])
  return Q.T

def computeQHatSeq(qs, rs, k, r):
  assert len(qs) == len(rs)
  #outQ = expr.zeros((r * len(rs), k)).force()
  if len(qs) == 1:
    return qs[0], rs[0]
  
  n_rows = 0
  for q in qs:
    n_rows += q.shape[1]

  outQ = np.zeros((n_rows, k))
  s = 0
  for i in range(0, len(qs)):
    Q = qs[i]
    Q = computeQtHat(Q, s, rs)
    outQ[i * r : i * r + Q.shape[0], :] = Q

    if s == 1:
      mergeR(rs[0], rs[1], k)
      del rs[1]
    else:
      s += 1
  
  assert len(rs) == 1
  return outQ, rs[0]


def _tree_reduce(Q, R, r):
  """Tree reduction. Given a list of Qs and Rs(Here we store them 
  into a spartan distributed array to save the network traiffic), 
  do a tree reduction to reduce them into a single Q and R.
  """
  n_tiles = Q.shape[1] / r
  qr = r
  round = 1
  while n_tiles > 1:
    t = int((n_tiles + 1) / 2)
    # tmp_array is only used for deciding how many workers we
    # launch for this tree reduction and tell the worker the index
    # of it.
    tmp_array = expr.zeros((t, 1), tile_hint=(1, 1)).force()
    tmp_array.foreach_tile(mapper_fn = _reduce_mapper,
                           kw = { 'Q' : Q,
                                  'R' : R,
                                  'r' : r,
                                  'round' : round})
    n_tiles = t 
    round += 1

  return Q.glom().T, R[0:R.shape[1]]


def qr(Y):
  '''
  Compute the QR decomposition of Y.

  For efficient operation, Y should be composed of contiguous row tiles.
  '''
  k = Y.shape[1]
  r = Y.tile_shape()[0]
  global_Q = expr.zeros((k, Y.shape[0])).force()
  global_R = expr.zeros(((Y.shape[0] + r - 1) / r * k, k)).force()

  results = Y.foreach_tile(mapper_fn = _qr_mapper,
                           kw = {'Y' : Y,
                                 'k' : k,
                                 'r' : r,
                                 'global_R' : global_R,
                                 'global_Q' : global_Q})

  return _tree_reduce(global_Q, global_R, r)
