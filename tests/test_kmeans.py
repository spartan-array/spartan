from spartan.examples import kmeans
import test_common


N_PTS = 10*10
N_CENTERS = 10
N_DIM = 5

class TestKmeans(test_common.ClusterTest):
  TILE_SIZE = 10
  def test_kmeans_expr(self):
    kmeans.run(N_PTS, N_CENTERS, N_DIM)