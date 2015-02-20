import test_common
from test_common import millis
from spartan.examples.spectral_clustering import spectral_cluster
from spartan import expr
from datetime import datetime


def benchmark_spectral_clustering(ctx, timer):
  #N_PTS = 500 * ctx.num_workers
  N_PTS = 50 * 64
  N_DIM = 2
  ITER = 5
  N_CENTERS = 5

  pts = expr.rand(N_PTS, N_DIM,
                  tile_hint=(N_PTS / ctx.num_workers, N_DIM)).evaluate()

  t1 = datetime.now()
  cluster_result = spectral_cluster(pts, N_CENTERS, ITER).glom()
  t2 = datetime.now()
  print 'spectral_cluster time:%s ms' % millis(t1, t2)

if __name__ == '__main__':
  test_common.run(__file__)
