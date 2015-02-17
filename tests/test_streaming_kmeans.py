import test_common
from test_common import millis
from spartan.examples.streaming_kmeans import streaming_kmeans
from spartan import expr
from datetime import datetime


def benchmark_streaming_kmeans(ctx, timer):
  #N_PTS = 100 * ctx.num_workers
  N_PTS = 100 * 64
  N_DIM = 2
  N_CENTERS = 5

  pts = expr.rand(N_PTS, N_DIM,
                  tile_hint=(N_PTS / ctx.num_workers, N_DIM)).evaluate()

  print pts.glom()
  t1 = datetime.now()
  cluster_result = streaming_kmeans(pts, k=N_CENTERS).glom()
  t2 = datetime.now()
  #print cluster_result.glom()
  time_cost = millis(t1, t2)
  print 'streaming_kmeans_cluster time:%s ms' % time_cost

if __name__ == '__main__':
  test_common.run(__file__)
