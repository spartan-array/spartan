import test_common
import spartan
from spartan.examples.sklearn.cluster import KMeans
from spartan import expr
from spartan.util import divup
from spartan.config import FLAGS
from datetime import datetime
from test_common import millis

N_PTS = 10*10
N_CENTERS = 10
N_DIM = 5
ITER = 5

class TestKmeans(test_common.ClusterTest):
  def test_kmeans_expr(self):
    FLAGS.opt_parakeet_gen = 0
    pts = expr.rand(N_PTS, N_DIM)
    k = KMeans(N_CENTERS, ITER)
    k.fit(pts)

def benchmark_kmeans(ctx, timer):
  print "#worker:", ctx.num_workers
  N_PTS = 1000 * 256
  N_CENTERS = 10
  N_DIM = 512
  ITER = 1
  pts = expr.rand(N_PTS, N_DIM)
  k = KMeans(N_CENTERS, ITER)
  t1 = datetime.now()
  k.fit(pts)
  t2 = datetime.now()
  cost_time = millis(t1, t2)
  print "total cost time:%s ms, per iter cost time:%s ms" % (cost_time, cost_time/ITER)

if __name__ == '__main__':
  test_common.run(__file__)
