import test_common
from test_common import millis
from spartan.examples.canopy_clustering import canopy_cluster
from spartan import expr
from datetime import datetime


def benchmark_canopy_clustering(ctx, timer):
  #N_PTS = 60000 * ctx.num_workers
  N_PTS = 30000 * 64
  N_DIM = 2

  pts = expr.rand(N_PTS, N_DIM,
                  tile_hint=(N_PTS / ctx.num_workers, N_DIM)).evaluate()

  t1 = datetime.now()
  cluster_result = canopy_cluster(pts).evaluate()
  t2 = datetime.now()
  print 'canopy_cluster time:%s ms' % millis(t1, t2)

if __name__ == '__main__':
  test_common.run(__file__)
