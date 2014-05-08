import test_common
from test_common import millis
from spartan.examples.fuzzy_kmeans import fuzzy_kmeans
from spartan import expr
from datetime import datetime    

def benchmark_fuzzy_kmeans(ctx, timer):
  #N_PTS = 40000 * ctx.num_workers
  N_PTS = 20000 * 64
  N_DIM = 2
  ITER = 5
  N_CENTERS = 10
  
  pts = expr.rand(N_PTS, N_DIM,
                  tile_hint=(N_PTS / ctx.num_workers, N_DIM)).force()

  t1 = datetime.now()
  cluster_result = fuzzy_kmeans(pts, k=N_CENTERS, num_iter=ITER).force()
  t2 = datetime.now()
  time_cost = millis(t1, t2)
  print 'fuzzy_cluster time:%s ms, per_iter:%s ms' % (time_cost, time_cost/ITER)
  
if __name__ == '__main__':
  test_common.run(__file__)
