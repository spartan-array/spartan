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
  
def _solve_U_or_M_mapper(array, ex, U_or_M, la, alpha, implicit_feedback):
  '''
  given A and U (or M), solve M (or U) such that A = U M' 
  using alternating least-squares factorization method
  
  Args:
    array(DistArray): the user-item (or item-user) rating matrix.
    ex(Extent): region being processed.
    U_or_M(DistArray): the matrix U (or M).
    la(float): the parameter of the als.
    alpha(int): confidence parameter used on implicit feedback.
    implicit_feedback(bool): whether using implicit_feedback method for als.
  '''
  rating_matrix = array.fetch(ex)
  U_or_M = U_or_M[:]
  
  if implicit_feedback:
    Y = U_or_M
    YT = Y.T
    YTY = np.dot(YT, Y)
 
  result = np.zeros((rating_matrix.shape[0], U_or_M.shape[1]))
  for i in range(rating_matrix.shape[0]):
    if implicit_feedback:
      result[i] = _implicit_feedback_als_solver(rating_matrix[i], la, alpha, Y, YT, YTY)
    else:
      non_zero_idx = rating_matrix[i].nonzero()[0]
      rating_vector = rating_matrix[i, non_zero_idx]
      feature_vectors = U_or_M[non_zero_idx]
      result[i] = _als_solver(feature_vectors, rating_vector, la)
    
  yield extent.create((ex.ul[0], 0), (ex.lr[0], U_or_M.shape[1]), (array.shape[0], U_or_M.shape[1])), result

def _init_M_mapper(array, ex, avg_rating):
  '''
  Initialize the M matrix with its first column equals to avg_rating.
  
  Args:
    array(DistArray): the array to be created.
    ex(Extent): region being processed.
    avg_rating(DistArray): the average rating for each item.
  '''
  avg_rating = avg_rating.fetch(extent.create((ex.ul[0],), (ex.lr[0],), avg_rating.shape))
  M = np.zeros(ex.shape)
  for i in avg_rating.nonzero()[0]:
    M[i, 0] = avg_rating[i]
    M[i, 1:] = np.random.rand(M.shape[1]-1)
  yield ex, M
  
def _transpose_mapper(array, ex, orig_array):
  '''
  Transpose ``orig_array`` into ``array``.
  
  Args:
    array(DistArray): destination array.
    ex(Extent): region being processed.
    orig_array(DistArray): array to be transposed.
  '''
  orig_ex = extent.create(ex.ul[::-1], ex.lr[::-1], orig_array.shape)
  yield ex, orig_array.fetch(orig_ex).transpose()
  
def als(A, la=0.065, alpha=40, implicit_feedback=False, num_features=20, num_iter=10):
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
  A = expr.force(A)
  AT = expr.shuffle(expr.ndarray((A.shape[1], A.shape[0]), dtype=A.dtype,
                                 tile_hint=(A.shape[1] / len(A.tiles), A.shape[0])),
                    _transpose_mapper, kw={'orig_array': A})
  
  num_items = A.shape[1]
  
  avg_rating = expr.sum(A, axis=0, tile_hint=(num_items / len(A.tiles),)) * 1.0 / \
               expr.count_nonzero(A, axis=0, tile_hint=(num_items / len(A.tiles),))
  
  M = expr.shuffle(expr.ndarray((num_items, num_features), 
                                tile_hint=(num_items / len(A.tiles), num_features)), 
                   _init_M_mapper, kw={'avg_rating': avg_rating})
  #util.log_warn('avg_rating:%s M:%s', avg_rating.glom(), M.glom())
  
  for i in range(num_iter):
    # Recomputing U
    U = expr.shuffle(A, _solve_U_or_M_mapper, kw={'U_or_M': M, 'la': la, 'alpha': alpha, 'implicit_feedback': implicit_feedback})
    # Recomputing M
    M = expr.shuffle(AT, _solve_U_or_M_mapper, kw={'U_or_M': U, 'la': la, 'alpha': alpha, 'implicit_feedback': implicit_feedback})
    
  return U, M
