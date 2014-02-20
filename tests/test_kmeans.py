import test_common
import spartan
from spartan.examples.sklearn.cluster import KMeans
from spartan import expr
from spartan.util import divup

N_PTS = 10*10
N_CENTERS = 10
N_DIM = 5
ITER = 5

class TestKmeans(test_common.ClusterTest):
  def test_kmeans_expr(self):
    ctx = spartan.blob_ctx.get()
    pts = expr.rand(N_PTS, N_DIM,
                  tile_hint=(divup(N_PTS, ctx.num_workers), N_DIM)).force()

    k = KMeans(N_CENTERS, ITER)
    k.fit(pts)
