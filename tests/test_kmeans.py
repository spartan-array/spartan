from spartan.dense import distarray
from spartan.examples import kmeans
from test_common import with_ctx

N_PTS = 10*10
N_CENTERS = 10
N_DIM = 5

distarray.TILE_SIZE = 5


@with_ctx
def test_kmeans_expr(ctx):
  kmeans.run(N_PTS, N_CENTERS, N_DIM)