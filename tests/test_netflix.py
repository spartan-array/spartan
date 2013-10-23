from nose.tools import nottest
from spartan import util
from spartan.examples import netflix
from spartan.util import Assert, divup
import math
import numpy as np
import spartan
import test_common


import pyximport; pyximport.install()

r = 100
d = 32

# how much to scale up matrix size by.
NUM_MOVIES = 2649429
NUM_USERS = 17770
P_RATING = 1e8 / (NUM_USERS * NUM_MOVIES)

SCALE = 2

U = NUM_USERS * SCALE
M = NUM_MOVIES * SCALE

def benchmark_netflix_sgd(ctx, timer):
  V = spartan.ndarray((U, M),
                      tile_hint=(divup(U, d), divup(M, d)))
  
  Mfactor = spartan.eager(spartan.rand(M, r))
  Ufactor = spartan.eager(spartan.rand(U, r))
  
#   V = spartan.map_extents(V, netflix.load_netflix_mapper,
#                           kw={ 'load_file' : '/big1/netflix.zip' })  
  
  V = timer.time_op('prep', lambda: spartan.eager(
        spartan.map_extents(V, netflix.fake_netflix_mapper, 
                          target=V, kw = { 'p_rating' : P_RATING })))
  
  for i in range(2):
    util.log_info('%d', i)
    _ = netflix.sgd(V, Mfactor, Ufactor)
    timer.time_op('netflix', lambda: _.force())
    

@nottest
def test_sgd_inner():
  N_ENTRIES = 2 * 1000 * 1000
  rows = np.random.randint(0, U, N_ENTRIES).astype(np.int64)
  cols = np.random.randint(0, M, N_ENTRIES).astype(np.int64)
  
  vals = np.random.randint(0, 5, N_ENTRIES).astype(np.float)
  u = np.random.randn(U, r)
  m = np.random.randn(M, r)
  
  for i in range(5):
    util.timeit(lambda: netflix._sgd_inner(rows, cols, vals, u, m), 'sgd')
  
if __name__ == '__main__':
  test_common.run(__file__)