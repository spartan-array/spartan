from spartan import util
import spartan
from spartan.examples import netflix
from spartan.util import divup


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
    
