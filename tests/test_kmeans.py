from spartan import expr
from spartan.dense import distarray
from test_common import with_ctx

N_PTS = 10*10
N_CENTERS = 10
N_DIM = 5

distarray.TILE_SIZE = 10

@with_ctx
def test_kmeans_expr(ctx):
  pts = expr.rand(N_PTS, N_DIM)
  centers = expr.rand(N_CENTERS, N_DIM)
  
