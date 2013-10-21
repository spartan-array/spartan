from spartan import util
from spartan.util import Assert, divup
from test_common import with_ctx
import netflix
import numpy as np
import spartan

d = 20
r = 1
P_RATING = 1e8 / (netflix.N_USERS * netflix.N_MOVIES)  

@with_ctx
def test_netflix(ctx):
  V = spartan.ndarray((netflix.N_USERS, netflix.N_MOVIES),
                      tile_hint=(divup(netflix.N_USERS, d), divup(netflix.N_MOVIES, d)))
  
  M = spartan.rand(netflix.N_MOVIES, r).force()
  U = spartan.rand(netflix.N_USERS, r).force()
  
#   V = spartan.map_extents(V, netflix.load_netflix_mapper,
#                           kw={ 'load_file' : '/big1/netflix.zip' })  
  
  V = spartan.eager(
        spartan.map_extents(V, netflix.fake_netflix_mapper, 
                          target=V, kw = { 'p_rating' : P_RATING }))
  
  for i in range(5):
    util.log_info('%d', i)
    _ = spartan.map_extents(V, netflix.sgd_netflix_mapper,
                        kw = { 'V' : V,
                               'M' : M,
                               'U' : U })
    _.force()
    
if __name__ == '__main__':
  test_netflix()