from spartan import util
from spartan.dense import extent
import cPickle
import numpy as np
import random
import scipy.sparse
import spartan
import zipfile

N_MOVIES = 2649429
N_USERS = 17770
EPSILON = 1e-6

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
  yield ex, data
  

def fake_netflix_mapper(inputs, ex, p_rating=None):
  row_start, row_end = ex.ul[0], ex.lr[0]
  col_start, col_end = ex.ul[1], ex.lr[1]
  
  data = scipy.sparse.dok_matrix(ex.shape, dtype=np.float)
  n_ratings = int(max(1, ex.size * p_rating))
  
  for i in range(n_ratings):
    uid = random.randrange(0, ex.shape[0])
    mid = random.randrange(0, ex.shape[1])
    data[uid, mid] = random.randint(0, 5)
  
  util.log_info('Loaded: %s', ex)
  yield ex, data

def sgd_netflix_mapper(inputs, ex, V=None, M=None, U=None):
  v = V.fetch(ex)
  
  
  u = U.select(ex[0].to_slice()) # (ex.shape[0] * r)
  m = M.select(ex[1].to_slice()) # (ex.shape[1] * r)
  
  #util.log_info('%s %s', m.shape, u.shape) 
  
  row_start, row_end = ex.ul[0], ex.lr[0]
  col_start, col_end = ex.ul[1], ex.lr[1]
  
  for (offset, mid), rating in v.iteritems():
    u_idx = offset
    m_idx = mid
    
    guess = u[u_idx].dot(m[m_idx].T)
    diff = rating - guess
    u[u_idx] = u[u_idx] + diff * EPSILON
    m[m_idx] = m[m_idx] + diff * EPSILON
    
  U.update_slice(ex[0].to_slice(), u)
  M.update_slice(ex[1].to_slice(), m)