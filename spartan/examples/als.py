import numpy as np
from scipy.linalg import lstsq
from spartan import expr, util
from spartan.array import extent


def _als_solver(feature_vectors, rating_vector, la):
  '''
  local alternating least-squares solver

  Args:
    feature_vectors(numpy.ndarray): part of the U or M matrix.
    rating_vector(numpy.ndarray): nonzero part of the rating vector of a user (or item).
    la(float): the parameter of the als.
  '''
  fvT = feature_vectors.T

  # compute Ai = fvT * fv + lambda * nui * E
  Ai = np.dot(fvT, feature_vectors) + la * rating_vector.shape[0] * np.eye(feature_vectors.shape[1])
  # compute Vi = fvT * rv.T
  Vi = np.dot(fvT, rating_vector)

  # compute Ai * ui = Vi
  return lstsq(Ai, Vi)[0]


def _implicit_feedback_als_solver(rating_vector, la, alpha, Y, YT, YTY):
  '''
  local implicit feedback alternating least-squares solver

  Args:
    rating_vector(numpy.ndarray): the rating vector of a user (or item).
    la(float): the parameter of the als.
    alpha(int): confidence parameter used on implicit feedback.
    Y(numpy.ndarray): the matrix U (or M).
    YT(numpy.ndarray): the transpose of matrix U (or M).
    YTY(numpy.ndarray): the result of YT dot Y.
  '''
  # Y' (Cu - I) Y + la * I
  Cu = rating_vector.reshape((rating_vector.shape[0], 1)) * alpha + 1
  YT_CuMinusI_Y_laI = np.dot(YT, Y * (Cu - 1)) + np.eye(Y.shape[1]) * la

  # Y' Cu p(u)
  YT_Cu_Pu = (Y * Cu)[rating_vector > 0].sum(axis=0)

  return lstsq(YTY + YT_CuMinusI_Y_laI, YT_Cu_Pu)[0]


def _solve_U_or_M_mapper(ex_a, rating_matrix, ex_b, U_or_M, la, alpha, implicit_feedback, shape=None):
  '''
  given A and U (or M), solve M (or U) such that A = U M'
  using alternating least-squares factorization method

  Args:
    rating_matrix: the user-item (or item-user) rating matrix.
    U_or_M: the matrix U (or M).
    la(float): the parameter of the als.
    alpha(int): confidence parameter used on implicit feedback.
    implicit_feedback(bool): whether using implicit_feedback method for als.
  '''
  if implicit_feedback:
    Y = U_or_M
    YT = Y.T
    YTY = np.dot(YT, Y)

  result = np.zeros((rating_matrix.shape[0], U_or_M.shape[1]))
  if implicit_feedback:
    for i in range(rating_matrix.shape[0]):
      result[i] = _implicit_feedback_als_solver(rating_matrix[i], la, alpha, Y, YT, YTY)
  else:
    for i in range(rating_matrix.shape[0]):
      non_zero_idx = rating_matrix[i].nonzero()[0]
      rating_vector = rating_matrix[i, non_zero_idx]
      feature_vectors = U_or_M[non_zero_idx]
      result[i] = _als_solver(feature_vectors, rating_vector, la)

  target_ex = extent.create((ex_a.ul[0], 0), (ex_a.lr[0], U_or_M.shape[1]), shape)
  yield target_ex, result


def als(A, la=0.065, alpha=40, implicit_feedback=False, num_features=20, num_iter=10, M=None):
  '''
  compute the factorization A = U M' using the alternating least-squares (ALS) method.

  where `A` is the "ratings" matrix which maps from a user and item to a rating score,
        `U` and `M` are the factor matrices, which represent user and item preferences.
  Args:
    A(Expr or DistArray): the rating matrix which maps from a user and item to a rating score.
    la(float): the parameter of the als.
    alpha(int): confidence parameter used on implicit feedback.
    implicit_feedback(bool): whether using implicit_feedback method for als.
    num_features(int): dimension of the feature space.
    num_iter(int): max iteration to run.
  '''
  num_users = A.shape[0]
  num_items = A.shape[1]

  AT = expr.transpose(A)

  avg_rating = expr.sum(A, axis=0) * 1.0 / expr.count_nonzero(A, axis=0)

  M = expr.rand(num_items, num_features)
  M = expr.assign(M, np.s_[:, 0], avg_rating.reshape((avg_rating.shape[0], 1)))

  #A = expr.retile(A, tile_hint=util.calc_tile_hint(A, axis=0))
  #AT = expr.retile(AT, tile_hint=util.calc_tile_hint(AT, axis=0))
  for i in range(num_iter):
    # Recomputing U
    shape = (num_users, num_features)
    U = expr.outer((A, M), (0, None), fn=_solve_U_or_M_mapper,
                   fn_kw={'la': la, 'alpha': alpha,
                          'implicit_feedback': implicit_feedback, 'shape': shape},
                   shape=shape, dtype=np.float)
    # Recomputing M
    shape = (num_items, num_features)
    M = expr.outer((AT, U), (0, None), fn=_solve_U_or_M_mapper,
                   fn_kw={'la': la, 'alpha': alpha,
                          'implicit_feedback': implicit_feedback, 'shape': shape},
                   shape=shape, dtype=np.float)
  return U, M
