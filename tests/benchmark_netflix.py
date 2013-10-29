from spartan import util
import spartan
from spartan.examples import netflix
from spartan.util import divup
import test_common


r = 100
d = 64

# how much to scale up matrix size by.
NUM_MOVIES = 2649429
NUM_USERS = 17770
P_RATING = 1e8 / (NUM_USERS * NUM_MOVIES)

SCALE = 1

U = NUM_USERS * SCALE
M = NUM_MOVIES * SCALE

def benchmark_netflix_sgd(ctx, timer):
  V = spartan.ndarray((U, M),
                      tile_hint=(divup(U, d), divup(M, d)))

  V = timer.time_op('prep', lambda: spartan.eager(
        spartan.map_extents(V, netflix.fake_netflix_mapper, 
                          target=V, kw = { 'p_rating' : P_RATING })))
 
#   V = spartan.map_extents(V, netflix.load_netflix_mapper,
#                           kw={ 'load_file' : '/big1/netflix.zip' })  

  for r in [25,50,100,200,400]:
    Mfactor = spartan.eager(spartan.rand(M, r, dtype=np.float32, tile_hint=(divup(M, d), r)))
    Ufactor = spartan.eager(spartan.rand(U, r, dtype=np.float32, tile_hint=(divup(U, d), r)))
  
    _ = netflix.sgd(V, Mfactor, Ufactor)
    timer.time_op('rank %d' % r, lambda: _.force())

if __name__ == '__main__':
  test_common.run(__file__)
