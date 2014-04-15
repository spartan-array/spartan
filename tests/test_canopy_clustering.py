import test_common
from test_common import millis
from spartan.examples.canopy_clustering import canopy_cluster
from spartan import expr
from datetime import datetime    

def benchmark_canopy_clustering(ctx, timer):
  N_PTS = 100 * ctx.num_workers
  N_DIM = 2

  pts = expr.rand(N_PTS, N_DIM,
                  tile_hint=(N_PTS / ctx.num_workers, N_DIM)).force()

  print pts.glom()
  t1 = datetime.now()
  cluster_result = canopy_cluster(pts)
  t2 = datetime.now()
  print 'canopy_cluster time:%s ms' % millis(t1, t2)
  
if __name__ == '__main__':
  test_common.run(__file__)