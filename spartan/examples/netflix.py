from copy import copy
from spartan import util, expr
from spartan.dense import extent
from spartan.expr import NotShapeable, lazify, make_primitive
from treelike import node
import cPickle
import numpy as np
import random
import scipy.sparse
import spartan
import time
import zipfile


#import parakeet
# EPSILON = 1e-5
# @util.synchronized
# @parakeet.jit
# def _sgd_inner(rows, cols, vals, u, m):
#   for offset, mid, rating in zip(rows, cols, vals):
#     u_idx = offset
#     m_idx = mid
#     guess = np.dot(u[u_idx], m[m_idx].T)
#     diff = rating - guess
#     u[u_idx] += u[u_idx] * diff * EPSILON
#     m[m_idx] += u[u_idx] * diff * EPSILON
#   

from netflix_core import _sgd_inner
FILE_START = 1

# utility functions for computing Netflix matrix factorization:
# r: rank of approximation (1~500)
# V: (N_USERS, N_MOVIES)
# M: (N_MOVIES, r)
# U: (N_USERS, r)
# V ~= UM

def load_netflix_mapper(inputs, ex, load_file=None):
  # first column will load all of the data
  row_start, row_end = ex.ul[0], ex.lr[0]
  col_start, col_end = ex.ul[1], ex.lr[1]
  
  data = scipy.sparse.dok_matrix(ex.shape, dtype=np.float)
  zf = zipfile.ZipFile(load_file, 'r', allowZip64=True)
  
  for i in range(row_start, row_end):
    offset = i - row_start
    row_data = cPickle.loads(zf.read('%d' % (i + FILE_START)))
    filtered = row_data[row_data['userid'] > col_start]
    filtered = filtered[filtered['userid'] < col_end]
    
    for uid, rating in filtered:
      uid -= col_start
      data[(offset, uid)] = rating
  
  util.log_info('Loaded: %s', ex)
  yield ex, data.tocoo()
  

def fake_netflix_mapper(inputs, ex, p_rating=None):
  '''
  Create "Netflix-like" data for the given extent.
  
  :param p_rating: Sparsity factor (probability a given cell will have a rating)
  '''
  n_ratings = int(max(1, ex.size * p_rating))
  
  uids = np.random.randint(0, ex.shape[0], n_ratings)
  mids = np.random.randint(0, ex.shape[1], n_ratings)
  ratings = np.random.randint(0, 5, n_ratings).astype(np.float32)

  data = scipy.sparse.coo_matrix((ratings, (uids, mids)), shape=ex.shape)
  yield ex, data
  
def sgd_netflix_mapper(inputs, ex, V=None, M=None, U=None, worklist=None):
  if not ex in worklist:
    return
  
  v = V.fetch(ex)
  
  u = U.select(ex[0].to_slice()) # size: (ex.shape[0] * r)
  m = M.select(ex[1].to_slice()) # size: (ex.shape[1] * r)
  
  _sgd_inner(v.row.astype(np.int64), 
             v.col.astype(np.int64), 
             v.data, u, m)
  
  U.update_slice(ex[0].to_slice(), u)
  M.update_slice(ex[1].to_slice(), m)

def strata_overlap(extents, v):
  for ex in extents:
    if v.ul[0] <= ex.ul[0] and v.lr[0] > ex.ul[0]: return True
    if v.ul[1] <= ex.ul[1] and v.lr[1] > ex.ul[1]: return True
  return False

def _compute_strata(V):
  strata = []
  extents = V.extents.keys() 
  random.shuffle(extents)
  
  while extents:
    stratum = []
    for ex in list(extents):
      if not strata_overlap(stratum, ex):
        stratum.append(ex)
    for ex in stratum:
      extents.remove(ex)  
    
    strata.append(stratum)
  
  return strata
  
def _evaluate_netflix(ctx, V, M, U):
  strata = _compute_strata(V)
  util.log_info('Start eval')
  for i, stratum in enumerate(strata):
    util.log_info('Processing stratum: %d of %d (size = %d)', i, len(strata), len(stratum))
    #for ex in stratum: print ex
    
    worklist = set(stratum)
    expr.map_extents(V, sgd_netflix_mapper, 
                     target=None,
                     kw={'V' : lazify(V), 'M' : lazify(M), 'U' : lazify(U),
                         'worklist' : worklist }).force()
  util.log_info('Eval done.')

NetflixSGD = make_primitive('NetflixSGD',
                            ['V', 'M', 'U'],
                            _evaluate_netflix)
    
def sgd(V, M, U):
  return NetflixSGD(V=V, M=M, U=U)
